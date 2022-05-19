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

from tvm.script import tir as T


@T.prim_func
def bsr(
    a: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
    block_size: T.int32,
) -> None:
    IO = T.dense_fixed(m)
    JO = T.sparse_variable(IO, (n, nnz), (indptr, indices), "int32")
    II = T.dense_fixed(block_size)
    JI = T.dense_fixed(block_size)
    A = T.match_sparse_buffer(a, (IO, JO, II, JI), "float32")
    T.evaluate(0)


@T.prim_func
def compressed_ell(
    a: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(0)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
    A = T.match_sparse_buffer(a, (O, I, J), "float32")
    T.evaluate(0)


@T.prim_func
def bsr_rewrite_with_preprocess(
    a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, feat_size: T.int32, nnz: T.int32, a_4: T.handle, indptr_4: T.handle, indices_4: T.handle, m_4: T.int32, n_4: T.int32, nnz_4: T.int32, a_16: T.handle, indptr_16: T.handle, indices_16: T.handle, m_16: T.int32, n_16: T.int32, nnz_16: T.int32, a_32: T.handle, indptr_32: T.handle, indices_32: T.handle, m_32: T.int32, n_32: T.int32, nnz_32: T.int32
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m, "int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n, "int32")
    K = T.dense_fixed(feat_size, "int32")
    IO_4 = T.dense_fixed(m_4, "int32")
    JO_4 = T.sparse_variable(IO_4, (n_4, nnz_4), (indptr_4, indices_4), "int32")
    II_4 = T.dense_fixed(4, "int32")
    JI_4 = T.dense_fixed(4, "int32")
    IO_16 = T.dense_fixed(m_16, "int32")
    JO_16 = T.sparse_variable(IO_16, (n_16, nnz_16), (indptr_16, indices_16), "int32")
    II_16 = T.dense_fixed(16, "int32")
    JI_16 = T.dense_fixed(16, "int32")
    IO_32 = T.dense_fixed(m_32, "int32")
    JO_32 = T.sparse_variable(IO_32, (n_32, nnz_32), (indptr_32, indices_32), "int32")
    II_32 = T.dense_fixed(32, "int32")
    JI_32 = T.dense_fixed(32, "int32")
    A = T.match_sparse_buffer(a, [I, J], dtype="float32")
    B = T.match_sparse_buffer(b, [J_detach, K], dtype="float32")
    C = T.match_sparse_buffer(c, [I, K], dtype="float32")
    A_4 = T.match_sparse_buffer(a_4, [IO_4, JO_4, II_4, JI_4], dtype="float32")
    A_16 = T.match_sparse_buffer(a_16, [IO_16, JO_16, II_16, JI_16], dtype="float32")
    A_32 = T.match_sparse_buffer(a_32, [IO_32, JO_32, II_32, JI_32], dtype="float32")
    # body
    # with T.block("root")
    with T.iter([IO_4, JO_4, II_4, JI_4], "SSSS", "rewrite_A_4") as [io_4, jo_4, ii_4, ji_4]:
        T.iter_attr({"preprocess": True})
        A_4[io_4, jo_4, ii_4, ji_4] = A[io_4 * 4 + ii_4, jo_4 * 4 + ji_4]
    with T.iter([IO_16, JO_16, II_16, JI_16], "SSSS", "rewrite_A_16") as [
        io_16,
        jo_16,
        ii_16,
        ji_16,
    ]:
        T.iter_attr({"preprocess": True})
        A_16[io_16, jo_16, ii_16, ji_16] = A[io_16 * 16 + ii_16, jo_16 * 16 + ji_16]
    with T.iter([IO_32, JO_32, II_32, JI_32], "SSSS", "rewrite_A_32") as [
        io_32,
        jo_32,
        ii_32,
        ji_32,
    ]:
        T.iter_attr({"preprocess": True})
        A_32[io_32, jo_32, ii_32, ji_32] = A[io_32 * 32 + ii_32, jo_32 * 32 + ji_32]
    with T.iter([IO_4, II_4, JO_4, JI_4, K], "SSRRS", "csrmm_4") as [io_4, ii_4, jo_4, ji_4, k]:
        with T.init():
            C[io_4 * 4 + ii_4, k] = T.float32(0)
        C[io_4 * 4 + ii_4, k] = (
            C[io_4 * 4 + ii_4, k] + A_4[io_4, jo_4, ii_4, ji_4] * B[jo_4 * 4 + ji_4, k]
        )
    with T.iter([IO_16, II_16, JO_16, JI_16, K], "SSRRS", "csrmm_16") as [
        io_16,
        ii_16,
        jo_16,
        ji_16,
        k,
    ]:
        with T.init():
            C[io_16 * 16 + ii_16, k] = T.float32(0)
        C[io_16 * 16 + ii_16, k] = (
            C[io_16 * 16 + ii_16, k] + A_16[io_16, jo_16, ii_16, ji_16] * B[jo_16 * 16 + ji_16, k]
        )
    with T.iter([IO_32, II_32, JO_32, JI_32, K], "SSRRS", "csrmm_32") as [
        io_32,
        ii_32,
        jo_32,
        ji_32,
        k,
    ]:
        with T.init():
            C[io_32 * 32 + ii_32, k] = T.float32(0)
        C[io_32 * 32 + ii_32, k] = (
            C[io_32 * 32 + ii_32, k] + A_32[io_32, jo_32, ii_32, ji_32] * B[jo_32 * 32 + ji_32, k]
        )
