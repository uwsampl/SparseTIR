/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file format.cc
 * \brief format conversion routine.
 */

#include <tvm/ir/expr.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>


namespace tvm {

using runtime::NDArray;

Array<Array<Array<NDArray>>> ColumnPartHyb(int num_rows, int num_cols, NDArray indptr,
                                           NDArray indices, int num_col_parts,
                                           Array<Integer> buckets) {
  int partition_size = (num_cols + num_col_parts - 1) / num_col_parts;
  int num_bkts = buckets.size();
  std::vector<int> buckets_vec;
  for (const Integer& bucket_size : buckets) {
    buckets_vec.push_back(bucket_size->value);
  }

  CHECK_EQ(indptr->dtype.bits, 32) << "Only support int32 index data type, got "
                                   << int(indptr->dtype.bits) << " bits for indptr.";
  CHECK_EQ(indices->dtype.bits, 32) << "Only support int32 index data type, got "
                                    << int(indices->dtype.bits) << " bits for indices.";
  CHECK_EQ(indptr->device.device_type, kDLCPU) << "Only support ColumnPartHyb conversion on CPU.";
  CHECK_EQ(indices->device.device_type, kDLCPU) << "Only support ColumnPartHyb conversion on CPU.";
  int* indptr_data = static_cast<int*>(indptr->data);
  int* indices_data = static_cast<int*>(indices->data);
  std::vector<std::unordered_multiset<int>> degree_counter(num_col_parts);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
      int row_id = i;
      int col_id = indices_data[j];
      int part_id = col_id / partition_size;
      degree_counter[part_id].insert(row_id);
    }
  }

  /* (num_parts, num_buckets, ...) */
  std::vector<std::vector<std::vector<int>>> row_indices(num_col_parts);
  std::vector<std::vector<std::vector<int>>> col_indices(num_col_parts);
  // init row_indices and col_indices
  for (int part_id = 0; part_id < num_col_parts; ++part_id) {
    for (int bucket_id = 0; bucket_id < num_bkts; ++bucket_id) {
      row_indices[part_id].push_back(std::vector<int>());
      col_indices[part_id].push_back(std::vector<int>());
    }
  }
  for (int i = 0; i < num_rows; ++i) {
    for (int j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
      int row_id = i;
      int col_id = indices_data[j];
      int part_id = col_id / partition_size;
      int degree = degree_counter[part_id].count(row_id);
      int bucket_id = std::upper_bound(buckets_vec.begin(), buckets_vec.end(), degree - 1) -
                      buckets_vec.begin();
      if (bucket_id == num_bkts) {
        bucket_id--;
      }
      int bucket_size = buckets_vec[bucket_id];
      bool create_new_bucket = false;
      int remainder = col_indices[part_id][bucket_id].size() % bucket_size;
      if (remainder != 0) {
        if (row_id != row_indices[part_id][bucket_id].back()) {
          // padding
          for (int k = remainder; k < bucket_size; ++k) {
            col_indices[part_id][bucket_id].push_back(0);
          }
          create_new_bucket = true;
        }
      } else {
        create_new_bucket = true;
      }
      if (create_new_bucket) {
        ICHECK(col_indices[part_id][bucket_id].size() % bucket_size == 0) << "Invalid padding";
        row_indices[part_id][bucket_id].push_back(row_id);
      }
      col_indices[part_id][bucket_id].push_back(col_id);
    }
  }

  // final padding and conversion to NDArray
  Array<Array<NDArray>> row_indices_nd;
  Array<Array<NDArray>> col_indices_nd;
  for (int part_id = 0; part_id < num_col_parts; ++part_id) {
    Array<NDArray> row_indices_part_local;
    Array<NDArray> col_indices_part_local;
    for (int bucket_id = 0; bucket_id < num_bkts; ++bucket_id) {
      int bucket_size = buckets_vec[bucket_id];
      // padding
      int remainder = col_indices[part_id][bucket_id].size() % bucket_size;
      if (remainder != 0) {
        for (int k = remainder; k < bucket_size; ++k) {
          col_indices[part_id][bucket_id].push_back(0);
        }
      }
      // conversion to NDArray
      int nnz = row_indices[part_id][bucket_id].size();
      ICHECK(int(col_indices[part_id][bucket_id].size()) == nnz * bucket_size) << "Padding error.";
      NDArray row_indices_bucket_local = NDArray::Empty({nnz}, {kDLInt, 32, 1}, {kDLCPU, 0});
      NDArray col_indices_bucket_local =
          NDArray::Empty({nnz, bucket_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
      row_indices_bucket_local.CopyFromBytes(row_indices[part_id][bucket_id].data(),
                                             nnz * sizeof(int));
      col_indices_bucket_local.CopyFromBytes(col_indices[part_id][bucket_id].data(),
                                             nnz * bucket_size * sizeof(int));
      row_indices_part_local.push_back(row_indices_bucket_local);
      col_indices_part_local.push_back(col_indices_bucket_local);
    }
    row_indices_nd.push_back(row_indices_part_local);
    col_indices_nd.push_back(col_indices_part_local);
  }

  // convert to NDArray

  return {row_indices_nd, col_indices_nd};
}

Array<NDArray> ConDense(NDArray indptr, NDArray indices, int block_size) {
  // Check inputs
  CHECK_EQ(indptr->dtype.bits, 32) << "Only support int32 index data type, got "
                                   << int(indptr->dtype.bits) << " bits for indptr.";
  CHECK_EQ(indices->dtype.bits, 32) << "Only support int32 index data type, got "
                                    << int(indices->dtype.bits) << " bits for indices.";
  CHECK_EQ(indptr->device.device_type, kDLCPU) << "Only support ConDense conversion on CPU.";
  CHECK_EQ(indices->device.device_type, kDLCPU) << "Only support ConDense conversion on CPU.";
  // Get data from NDArrays
  int* indptr_data = static_cast<int*>(indptr->data);
  int* indices_data = static_cast<int*>(indices->data);
  // Set up return values
  int n = indptr->shape[0] - 1;
  int num_blocks = (n + block_size - 1) / block_size;
  std::vector<int> ret_indptr(num_blocks + 1);
  std::vector<int> ret_indices;
  ret_indptr[0] = 0;
  // Condense matrix
  for (int block_id = 0; block_id < num_blocks; block_id++) {
    int curr_block = block_id * block_size;
    int next_block = curr_block + block_size;
    int lo = indptr_data[curr_block];
    int hi = next_block > n ? indptr_data[n] : indptr_data[next_block];
    // Find unique indices from lo to hi
    std::vector<int> unique(hi - lo);
    for (int i = 0; i < hi - lo; i++) {
      unique[i] = indices_data[lo + i];
    }
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    ret_indices.insert(ret_indices.end(), unique.begin(), unique.end());
    ret_indptr[block_id + 1] = ret_indptr[block_id] + unique.size();
  }

  // Convert to NDArray
  int ret_indptr_size = ret_indptr.size();
  int ret_indices_size = ret_indices.size();
  NDArray indptr_nd = NDArray::Empty({ret_indptr_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
  NDArray indices_nd = NDArray::Empty({ret_indices_size}, {kDLInt, 32, 1}, {kDLCPU, 0});
  indptr_nd.CopyFromBytes(ret_indptr.data(), ret_indptr_size * sizeof(int));
  indices_nd.CopyFromBytes(ret_indices.data(), ret_indices_size * sizeof(int));
  return {indptr_nd, indices_nd};
}

namespace sparse {
TVM_REGISTER_GLOBAL("tir.sparse.ColumnPartHyb").set_body_typed(ColumnPartHyb);
TVM_REGISTER_GLOBAL("tir.sparse.ConDense").set_body_typed(ConDense);
}  // namespace sparse
}  // namespace tvm