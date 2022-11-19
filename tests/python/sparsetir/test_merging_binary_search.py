# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-function-docstring,missing-module-docstring
import tvm
from tvm.script import tir as T
from tvm.sparse import lower_sparse_iter


@T.prim_func
def func(indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, nnz: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices))
    A = T.alloc_sparse_buffer([I, J], "float32")
    with T.iter([T.fuse(I, J)], "SS", "test") as [vi, vj]:
        A[vi, vj] = (vi + vj) * (vi - vj)


@T.prim_func
def lowered(indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, nnz: T.int32) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 1})
    I = T.dense_fixed(m, idtype="int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), idtype="int32", sorted=True)
    J_dense = T.dense_variable(I, (n, nnz), indptr, idtype="int32")
    J_indptr = T.match_sparse_buffer(indptr, [I], dtype="int32", extra_storage=1)
    J_indices = T.match_sparse_buffer(indices, [I, J_dense], dtype="int32")
    # body
    # with T.block("root")
    A = T.alloc_sparse_buffer([I, J], dtype="float32", extra_storage=0)
    mid_0 = T.alloc_sparse_buffer([I, J], dtype="int32", extra_storage=0)
    T.assume_buffer_domain(J_indptr, [0, nnz])
    T.assume_buffer_domain(J_indices, [0, n])
    T.assume_buffer_domain(mid_0, [0, m])
    for vj in T.serial(nnz):
        with T.block("binary_search_block_0_0"):
            vvi = T.axis.spatial(1, 0)
            vvj = T.axis.spatial(nnz, vj)
            T.reads(J_indptr[0 : m + 1])
            T.writes(mid_0[vvi, vvj])
            T.block_attr({"preprocess": True, "sparse": True})
            low = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
            high = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
            low[0] = 0
            high[0] = m + 1
            mid_0[vvi, vvj] = low[0] + (high[0] - low[0]) // 2
            while low[0] < high[0]:
                if J_indptr[mid_0[vvi, vvj]] > vvj:
                    high[0] = mid_0[vvi, vvj]
                else:
                    low[0] = mid_0[vvi, vvj] + 1
                mid_0[vvi, vvj] = low[0] + (high[0] - low[0]) // 2
            mid_0[vvi, vvj] = mid_0[vvi, vvj] - 1
    for vj in T.serial(nnz):
        with T.block("test0"):
            vvi = T.axis.spatial(1, 0)
            vvj = T.axis.spatial(nnz, vj)
            T.reads(mid_0[vvi, vvj], J_indices[vvi, vvj])
            T.writes(A[vvi, vvj])
            T.block_attr({"sparse": True})
            A[vvi, vvj] = (mid_0[vvi, vvj] + J_indices[vvi, vvj]) * (
                mid_0[vvi, vvj] - J_indices[vvi, vvj]
            )


def test_merging_binary_search():
    mod = tvm.IRModule.from_expr(func)
    mod = lower_sparse_iter(mod)
    print(mod["main"].script())
    tvm.ir.assert_structural_equal(mod["main"], lowered)


if __name__ == "__main__":
    test_merging_binary_search()
