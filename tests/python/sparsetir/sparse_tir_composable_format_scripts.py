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
def padding(
    a: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz_chunks: T.int32,
    pad_size: T.int32,
) -> None:
    I = T.dense_fixed(m)
    JO = T.dense_variable(I, ((n + pad_size - 1) // pad_size, nnz_chunks), indptr, "int32")
    JI = T.sparse_fixed(JO, (n, pad_size), indices, "int32")
    A = T.match_sparse_buffer(a, (I, JO, JI), "float32")
    T.evaluate(0)


@T.prim_func
def ell(
    a: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    O = T.dense_fixed(1)
    I = T.sparse_variable(O, (m, num_rows), (indptr_i, indices_i))
    J = T.sparse_fixed(I, (n, nnz_cols), indices_j)
    A = T.match_sparse_buffer(a, (O, I, J), "float32")
    T.evaluate(0)


@T.prim_func
def ell3d(
    a: T.handle,
    indptr_io: T.handle,
    indices_ii: T.handle,
    indices_j: T.handle,
    d0: T.int32,
    d1: T.int32,
    d2: T.int32,
    nnz: T.int32,
    nnz_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    R = T.dense_fixed(d0, idtype="int32")
    IO = T.dense_variable(R, (d1, nnz), indptr_io, idtype="int32")
    II = T.sparse_fixed(IO, (d2, nnz_rows), indices_ii, idtype="int32")
    J = T.sparse_fixed(II, (d2, nnz_cols), indices_j, idtype="int32")
    A = T.match_sparse_buffer(a, (R, IO, II, J), dtype="float32")
    T.evaluate(0)


@T.prim_func
def ell3d_fp16(
    a: T.handle,
    wx: T.handle,
    indptr_io: T.handle,
    indices_ii: T.handle,
    indices_j: T.handle,
    d0: T.int32,
    d1: T.int32,
    d2: T.int32,
    nnz: T.int32,
    nnz_rows: T.int32,
    nnz_cols: T.int32,
    feat_size: T.int32,
) -> None:
    R = T.dense_fixed(d0, idtype="int32")
    IO = T.dense_variable(R, (d1, nnz), indptr_io, idtype="int32")
    II = T.sparse_fixed(IO, (d2, nnz_rows), indices_ii, idtype="int32")
    J = T.sparse_fixed(II, (d2, nnz_cols), indices_j, idtype="int32")
    FO = T.dense_fixed(feat_size, idtype="int32")
    A = T.match_sparse_buffer(a, (R, IO, II, J), dtype="float16")
    WX = T.match_sparse_buffer(wx, (R, IO, II, FO), dtype="float16")
    T.evaluate(0)


@T.prim_func
def bsr_rewrite_with_preprocess(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
    a_4: T.handle,
    indptr_4: T.handle,
    indices_4: T.handle,
    m_4: T.int32,
    n_4: T.int32,
    nnz_4: T.int32,
    a_16: T.handle,
    indptr_16: T.handle,
    indices_16: T.handle,
    m_16: T.int32,
    n_16: T.int32,
    nnz_16: T.int32,
    a_32: T.handle,
    indptr_32: T.handle,
    indices_32: T.handle,
    m_32: T.int32,
    n_32: T.int32,
    nnz_32: T.int32,
) -> None:
    # function attr dict
    T.func_attr(
        {"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2, "composable": 1}
    )
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


@T.prim_func
def ell_rewrite_with_preprocess(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
    a_4: T.handle,
    indptr_i_4: T.handle,
    indices_i_4: T.handle,
    indices_j_4: T.handle,
    m_4: T.int32,
    n_4: T.int32,
    num_rows_4: T.int32,
    a_8: T.handle,
    indptr_i_8: T.handle,
    indices_i_8: T.handle,
    indices_j_8: T.handle,
    m_8: T.int32,
    n_8: T.int32,
    num_rows_8: T.int32,
    a_16: T.handle,
    indptr_i_16: T.handle,
    indices_i_16: T.handle,
    indices_j_16: T.handle,
    m_16: T.int32,
    n_16: T.int32,
    num_rows_16: T.int32,
    a_32: T.handle,
    indptr_i_32: T.handle,
    indices_i_32: T.handle,
    indices_j_32: T.handle,
    m_32: T.int32,
    n_32: T.int32,
    num_rows_32: T.int32,
    a_64: T.handle,
    indptr_i_64: T.handle,
    indices_i_64: T.handle,
    indices_j_64: T.handle,
    m_64: T.int32,
    n_64: T.int32,
    num_rows_64: T.int32,
    a_128: T.handle,
    indptr_i_128: T.handle,
    indices_i_128: T.handle,
    indices_j_128: T.handle,
    m_128: T.int32,
    n_128: T.int32,
    num_rows_128: T.int32,
    a_512: T.handle,
    indptr_i_512: T.handle,
    indices_i_512: T.handle,
    indices_j_512: T.handle,
    m_512: T.int32,
    n_512: T.int32,
    num_rows_512: T.int32,
) -> None:
    # function attr dict
    T.func_attr(
        {"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2, "composable": 1}
    )
    I = T.dense_fixed(m, "int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n, "int32")
    K = T.dense_fixed(feat_size, "int32")
    O_4 = T.dense_fixed(1, "int32")
    I_4 = T.sparse_variable(O_4, (m_4, num_rows_4), (indptr_i_4, indices_i_4), "int32")
    J_4 = T.sparse_fixed(I_4, (n_4, 4), indices_j_4, "int32")
    O_8 = T.dense_fixed(1, "int32")
    I_8 = T.sparse_variable(O_8, (m_8, num_rows_8), (indptr_i_8, indices_i_8), "int32")
    J_8 = T.sparse_fixed(I_8, (n_8, 8), indices_j_8, "int32")
    O_16 = T.dense_fixed(1, "int32")
    I_16 = T.sparse_variable(O_16, (m_16, num_rows_16), (indptr_i_16, indices_i_16), "int32")
    J_16 = T.sparse_fixed(I_16, (n_16, 16), indices_j_16, "int32")
    O_32 = T.dense_fixed(1, "int32")
    I_32 = T.sparse_variable(O_32, (m_32, num_rows_32), (indptr_i_32, indices_i_32), "int32")
    J_32 = T.sparse_fixed(I_32, (n_32, 32), indices_j_32, "int32")
    O_64 = T.dense_fixed(1, "int32")
    I_64 = T.sparse_variable(O_64, (m_64, num_rows_64), (indptr_i_64, indices_i_64), "int32")
    J_64 = T.sparse_fixed(I_64, (n_64, 64), indices_j_64, "int32")
    O_128 = T.dense_fixed(1, "int32")
    I_128 = T.sparse_variable(O_128, (m_128, num_rows_128), (indptr_i_128, indices_i_128), "int32")
    J_128 = T.sparse_fixed(I_128, (n_128, 128), indices_j_128, "int32")
    O_512 = T.dense_fixed(1, "int32")
    I_512 = T.sparse_variable(O_512, (m_512, num_rows_512), (indptr_i_512, indices_i_512), "int32")
    J_512 = T.sparse_fixed(I_512, (n_512, 512), indices_j_512, "int32")
    A = T.match_sparse_buffer(a, [I, J], dtype="float32")
    B = T.match_sparse_buffer(b, [J_detach, K], dtype="float32")
    C = T.match_sparse_buffer(c, [I, K], dtype="float32")
    A_4 = T.match_sparse_buffer(a_4, [O_4, I_4, J_4], dtype="float32")
    A_8 = T.match_sparse_buffer(a_8, [O_8, I_8, J_8], dtype="float32")
    A_16 = T.match_sparse_buffer(a_16, [O_16, I_16, J_16], dtype="float32")
    A_32 = T.match_sparse_buffer(a_32, [O_32, I_32, J_32], dtype="float32")
    A_64 = T.match_sparse_buffer(a_64, [O_64, I_64, J_64], dtype="float32")
    A_128 = T.match_sparse_buffer(a_128, [O_128, I_128, J_128], dtype="float32")
    A_512 = T.match_sparse_buffer(a_512, [O_512, I_512, J_512], dtype="float32")
    # body
    # with T.block("root")
    with T.iter([O_4, I_4, J_4], "SSS", "rewrite_A_4") as [o_4, i_4, j_4]:
        T.iter_attr({"preprocess": True})
        A_4[o_4, i_4, j_4] = A[i_4, j_4]
    with T.iter([O_8, I_8, J_8], "SSS", "rewrite_A_8") as [o_8, i_8, j_8]:
        T.iter_attr({"preprocess": True})
        A_8[o_8, i_8, j_8] = A[i_8, j_8]
    with T.iter([O_16, I_16, J_16], "SSS", "rewrite_A_16") as [o_16, i_16, j_16]:
        T.iter_attr({"preprocess": True})
        A_16[o_16, i_16, j_16] = A[i_16, j_16]
    with T.iter([O_32, I_32, J_32], "SSS", "rewrite_A_32") as [o_32, i_32, j_32]:
        T.iter_attr({"preprocess": True})
        A_32[o_32, i_32, j_32] = A[i_32, j_32]
    with T.iter([O_64, I_64, J_64], "SSS", "rewrite_A_64") as [o_64, i_64, j_64]:
        T.iter_attr({"preprocess": True})
        A_64[o_64, i_64, j_64] = A[i_64, j_64]
    with T.iter([O_128, I_128, J_128], "SSS", "rewrite_A_128") as [o_128, i_128, j_128]:
        T.iter_attr({"preprocess": True})
        A_128[o_128, i_128, j_128] = A[i_128, j_128]
    with T.iter([O_512, I_512, J_512], "SSS", "rewrite_A_512") as [o_512, i_512, j_512]:
        T.iter_attr({"preprocess": True})
        A_512[o_512, i_512, j_512] = A[i_512, j_512]
    with T.iter([O_4, I_4, J_4, K], "SSRS", "csrmm_4") as [o_4, i_4, j_4, k]:
        with T.init():
            C[i_4, k] = T.float32(0)
        C[i_4, k] = C[i_4, k] + A_4[o_4, i_4, j_4] * B[j_4, k]
    with T.iter([O_8, I_8, J_8, K], "SSRS", "csrmm_8") as [o_8, i_8, j_8, k]:
        with T.init():
            C[i_8, k] = T.float32(0)
        C[i_8, k] = C[i_8, k] + A_8[o_8, i_8, j_8] * B[j_8, k]
    with T.iter([O_16, I_16, J_16, K], "SSRS", "csrmm_16") as [o_16, i_16, j_16, k]:
        with T.init():
            C[i_16, k] = T.float32(0)
        C[i_16, k] = C[i_16, k] + A_16[o_16, i_16, j_16] * B[j_16, k]
    with T.iter([O_32, I_32, J_32, K], "SSRS", "csrmm_32") as [o_32, i_32, j_32, k]:
        with T.init():
            C[i_32, k] = T.float32(0)
        C[i_32, k] = C[i_32, k] + A_32[o_32, i_32, j_32] * B[j_32, k]
    with T.iter([O_64, I_64, J_64, K], "SSRS", "csrmm_64") as [o_64, i_64, j_64, k]:
        with T.init():
            C[i_64, k] = T.float32(0)
        C[i_64, k] = C[i_64, k] + A_64[o_64, i_64, j_64] * B[j_64, k]
    with T.iter([O_128, I_128, J_128, K], "SSRS", "csrmm_128") as [o_128, i_128, j_128, k]:
        with T.init():
            C[i_128, k] = T.float32(0)
        C[i_128, k] = C[i_128, k] + A_128[o_128, i_128, j_128] * B[j_128, k]
    with T.iter([O_512, I_512, J_512, K], "SSRS", "csrmm_512") as [o_512, i_512, j_512, k]:
        with T.init():
            C[i_512, k] = T.float32(0)
        C[i_512, k] = C[i_512, k] + A_512[o_512, i_512, j_512] * B[j_512, k]


@T.prim_func
def padding_rewrite_with_preprocess(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
    a_32: T.handle,
    indptr_32: T.handle,
    indices_32: T.handle,
    m_32: T.int32,
    n_32: T.int32,
    nnz_chunks_32: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m, "int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n, "int32")
    K = T.dense_fixed(feat_size, "int32")
    I_32 = T.dense_fixed(m_32, "int32")
    JO_32 = T.dense_variable(I_32, ((n_32 + 32 - 1) // 32, nnz_chunks_32), indptr_32, "int32")
    JI_32 = T.sparse_fixed(JO_32, (n_32, 32), indices_32, "int32")
    A = T.match_sparse_buffer(a, [I, J], dtype="float32")
    B = T.match_sparse_buffer(b, [J_detach, K], dtype="float32")
    C = T.match_sparse_buffer(c, [I, K], dtype="float32")
    A_32 = T.match_sparse_buffer(a_32, [I_32, JO_32, JI_32], dtype="float32")
    # body
    # with T.block("root")
    with T.iter([I_32, JO_32, JI_32], "SSS", "rewrite_A_32") as [i_32, jo_32, ji_32]:
        T.iter_attr({"preprocess": True})
        A_32[i_32, jo_32, ji_32] = A[i_32, ji_32]
    with T.iter([I_32, JO_32, JI_32, K], "SRRS", "csrmm_32") as [i_32, jo_32, ji_32, k]:
        with T.init():
            C[i_32, k] = T.float32(0)
        C[i_32, k] = C[i_32, k] + A_32[i_32, jo_32, ji_32] * B[ji_32, k]
