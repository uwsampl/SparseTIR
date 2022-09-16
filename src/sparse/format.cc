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

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/registry.h>
#include <cassert>
#include <unordered_map>

namespace tvm {

using runtime::NDArray;


Array<Array<Array<NDArray>>> ColumnPartHyb(
  int num_rows,
  int num_cols,
  NDArray indptr,
  NDArray indices,
  int column_parts,
  Array<int> buckets
) {
  Array<Array<NDArray>> rst_row_indices;
  Array<Array<NDArray>> rst_col_indices;
  int partition_size = (num_cols + column_parts - 1) / column_parts;

  assert(indptr->dtype.bits == 32);
  assert(indices->dtype.bits == 32);  
  int* indptr_data = static_cast<int*>(indptr->data);
  int* indices_data = static_cast<int*>(indices->data);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
      int row_id = i;
      int col_id = indices_data[j];
      int part_id = col_id / partition_size;
    }
  }

  return {rst_row_indices, rst_col_indices};
}

namespace sparse {
  TVM_REGISTER_GLOBAL("tir.sparse.ColumnPartHyb").set_body_typed(ColumnPartHyb);
}  // namespace sparse
}  // namespace tvm