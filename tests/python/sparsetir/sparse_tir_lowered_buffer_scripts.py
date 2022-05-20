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
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nnz], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [n * feat_size], dtype="float32", strides=[1])
    C_data = T.match_buffer(c, [m * feat_size], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [m + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nnz], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for i in T.serial(m):
        with T.block("csrmm0"):
            vi = T.axis.spatial(m, i)
            T.reads(
                J_indptr_data[0 : m + 1],
                A_data[0:nnz],
                B_data[0 : n * feat_size],
                J_indices_data[0:nnz],
            )
            T.writes(C_data[0 : m * feat_size])
            T.block_attr({"sparse": True})
            for j, k in T.grid(J_indptr_data[vi + 1] - J_indptr_data[vi], feat_size):
                with T.block("csrmm1"):
                    vj = T.axis.reduce(n, j)
                    vk = T.axis.spatial(feat_size, k)
                    T.reads(
                        A_data[vj + J_indptr_data[vi]],
                        B_data[J_indices_data[vj + J_indptr_data[vi]] * feat_size + vk],
                        J_indices_data[vj + J_indptr_data[vi]],
                    )
                    T.writes(C_data[vi * feat_size + vk])
                    T.block_attr({"sparse": True})
                    with T.init():
                        C_data[vi * feat_size + vk] = T.float32(0)
                    C_data[vi * feat_size + vk] = (
                        C_data[vi * feat_size + vk]
                        + A_data[vj + J_indptr_data[vi]]
                        * B_data[J_indices_data[vj + J_indptr_data[vi]] * feat_size + vk]
                    )


@T.prim_func
def csrmm_dense_iter(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nnz], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [n * feat_size], dtype="float32", strides=[1])
    C_data = T.match_buffer(c, [m * feat_size], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [m + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nnz], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    mid_0_data = T.alloc_buffer([m * n], dtype="int32", strides=[1])
    for i, j in T.grid(m, n):
        with T.block("binary_search_block_0_0"):
            vi, vj = T.axis.remap("SR", [i, j])
            T.reads()
            T.writes()
            T.block_attr({"preprocess": True, "sparse": True})
            low = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
            high = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
            low[0] = 0
            high[0] = J_indptr_data[vi + 1] - J_indptr_data[vi]
            while low[0] < high[0]:
                mid_0_data[vi * n + vj] = low[0] + (high[0] - low[0]) // 2
                if J_indices_data[mid_0_data[vi * n + vj] + J_indptr_data[vi]] < vj:
                    low[0] = mid_0_data[vi * n + vj] + 1
                else:
                    high[0] = mid_0_data[vi * n + vj]
    for i, j, k in T.grid(m, n, feat_size):
        with T.block("csrmm0"):
            vi, vj, vk = T.axis.remap("SRS", [i, j, k])
            T.reads(
                A_data[mid_0_data[vi * n + vj] + J_indptr_data[vi]],
                mid_0_data[vi * n + vj],
                B_data[vj * feat_size + vk],
            )
            T.writes(C_data[vi * feat_size + vk])
            T.block_attr({"sparse": True})
            with T.init():
                C_data[vi * feat_size + vk] = T.float32(0)
            C_data[vi * feat_size + vk] = (
                C_data[vi * feat_size + vk]
                + A_data[mid_0_data[vi * n + vj] + J_indptr_data[vi]] * B_data[vj * feat_size + vk]
            )


@T.prim_func
def segment_reduce(a: T.handle, b: T.handle, indptr: T.handle, n: T.int32, nnz: T.int32) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nnz], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [n], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [n + 1], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for i in T.serial(n):
        with T.block("segment_reduce0"):
            vi = T.axis.spatial(n, i)
            T.reads(J_indptr_data[0 : n + 1], A_data[0:nnz])
            T.writes(B_data[vi])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("segment_reduce1"):
                    vj = T.axis.reduce(100, j)
                    T.reads(A_data[vj + J_indptr_data[vi]])
                    T.writes(B_data[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B_data[vi] = T.float32(0)
                    B_data[vi] = B_data[vi] + A_data[vj + J_indptr_data[vi]]


@T.prim_func
def csr_reduce(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nnz], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [n], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [n + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nnz], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for i in T.serial(n):
        with T.block("csr_reduce0"):
            vi = T.axis.spatial(n, i)
            T.reads(J_indptr_data[0 : n + 1], A_data[0:nnz])
            T.writes(B_data[vi])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("csr_reduce1"):
                    vj = T.axis.reduce(m, j)
                    T.reads(A_data[vj + J_indptr_data[vi]])
                    T.writes(B_data[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B_data[vi] = T.float32(0)
                    B_data[vi] = B_data[vi] + A_data[vj + J_indptr_data[vi]]


@T.prim_func
def bsrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    nnzb: T.int32,
    blk: T.int32,
    feat_size: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nnzb * blk * blk], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [mb * blk * feat_size], dtype="float32", strides=[1])
    C_data = T.match_buffer(c, [nb * blk * feat_size], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [nb + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nnzb], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for i, bi, bj, f in T.grid(nb, blk, blk, feat_size):
        with T.block("bsrmm0"):
            vi, vbi, vbj, vf = T.axis.remap("SSRS", [i, bi, bj, f])
            T.reads(
                J_indptr_data[0 : nb + 1],
                A_data[0 : nnzb * blk * blk],
                B_data[0 : mb * blk * feat_size],
                J_indices_data[0:nnzb],
            )
            T.writes(C_data[vi * (blk * feat_size) + vbi * feat_size + vf])
            T.block_attr({"sparse": True})
            with T.init():
                C_data[vi * (blk * feat_size) + vbi * feat_size + vf] = T.float32(0)
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("bsrmm1"):
                    vj = T.axis.reduce(mb, j)
                    T.reads(
                        A_data[(vj + J_indptr_data[vi]) * (blk * blk) + vbi * blk + vbj],
                        B_data[
                            J_indices_data[vj + J_indptr_data[vi]] * (blk * feat_size)
                            + vbj * feat_size
                            + vf
                        ],
                        J_indices_data[vj + J_indptr_data[vi]],
                    )
                    T.writes(C_data[vi * (blk * feat_size) + vbi * feat_size + vf])
                    T.block_attr({"sparse": True})
                    C_data[vi * (blk * feat_size) + vbi * feat_size + vf] = (
                        C_data[vi * (blk * feat_size) + vbi * feat_size + vf]
                        + A_data[(vj + J_indptr_data[vi]) * (blk * blk) + vbi * blk + vbj]
                        * B_data[
                            J_indices_data[vj + J_indptr_data[vi]] * (blk * feat_size)
                            + vbj * feat_size
                            + vf
                        ]
                    )


@T.prim_func
def ellmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    feat_size: T.int32,
    col: T.int32,
    blk: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nb * col * blk * blk], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [mb * blk * feat_size], dtype="float32", strides=[1])
    C_data = T.match_buffer(c, [nb * blk * feat_size], dtype="float32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nb * col], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for i, j, bi, bj, f in T.grid(nb, col, blk, blk, feat_size):
        with T.block("ellmm0"):
            vi, vj, vbi, vbj, vf = T.axis.remap("SRSRS", [i, j, bi, bj, f])
            T.reads(
                A_data[vi * (col * blk * blk) + vj * (blk * blk) + vbi * blk + vbj],
                B_data[J_indices_data[vi * col + vj] * (blk * feat_size) + vbj * feat_size + vf],
                J_indices_data[vi * col + vj],
            )
            T.writes(C_data[vi * (blk * feat_size) + vbi * feat_size + vf])
            T.block_attr({"sparse": True})
            with T.init():
                C_data[vi * (blk * feat_size) + vbi * feat_size + vf] = T.float32(0)
            C_data[vi * (blk * feat_size) + vbi * feat_size + vf] = (
                C_data[vi * (blk * feat_size) + vbi * feat_size + vf]
                + A_data[vi * (col * blk * blk) + vj * (blk * blk) + vbi * blk + vbj]
                * B_data[J_indices_data[vi * col + vj] * (blk * feat_size) + vbj * feat_size + vf]
            )


@T.prim_func
def csr_element_wise(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nnz], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [nnz], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [m + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nnz], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for i in T.serial(m):
        with T.block("csr_element_wise0"):
            vi = T.axis.spatial(m, i)
            T.reads(J_indptr_data[0 : m + 1], A_data[0:nnz])
            T.writes(B_data[0:nnz])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("csr_element_wise1"):
                    vj = T.axis.spatial(n, j)
                    T.reads(A_data[vj + J_indptr_data[vi]])
                    T.writes(B_data[vj + J_indptr_data[vi]])
                    T.block_attr({"sparse": True})
                    B_data[vj + J_indptr_data[vi]] = A_data[vj + J_indptr_data[vi]] * T.float32(2.5)


@T.prim_func
def hyper_gnn(
    x: T.handle,
    y: T.handle,
    indptr: T.handle,
    indices: T.handle,
    indptr_T: T.handle,
    indices_T: T.handle,
    n: T.int32,
    m: T.int32,
    nnz: T.int32,
    feat_size: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    X_data = T.match_buffer(x, [n * feat_size], dtype="float32", strides=[1])
    Y_data = T.match_buffer(y, [n * feat_size], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [n + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nnz], dtype="int32", strides=[1])
    I_T_indptr_data = T.match_buffer(indptr_T, [m + 1], dtype="int32", strides=[1])
    I_T_indices_data = T.match_buffer(indices_T, [nnz], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for i, f in T.grid(n, feat_size):
        with T.block("hyper_gnn0"):
            vi, vf = T.axis.remap("SS", [i, f])
            T.reads(
                J_indptr_data[0 : n + 1],
                I_T_indptr_data[0 : m + 1],
                J_indices_data[0:nnz],
                X_data[0 : n * feat_size],
                I_T_indices_data[0:nnz],
            )
            T.writes(Y_data[vi * feat_size + vf])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("hyper_gnn1"):
                    vj = T.axis.reduce(m, j)
                    T.reads(
                        I_T_indptr_data[0 : m + 1],
                        J_indices_data[vj + J_indptr_data[vi]],
                        X_data[0 : n * feat_size],
                        I_T_indices_data[0:nnz],
                    )
                    T.writes(Y_data[vi * feat_size + vf])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Y_data[vi * feat_size + vf] = T.float32(0)
                    for i_t in T.serial(
                        I_T_indptr_data[J_indices_data[vj + J_indptr_data[vi]] + 1]
                        - I_T_indptr_data[J_indices_data[vj + J_indptr_data[vi]]]
                    ):
                        with T.block("hyper_gnn2"):
                            vi_t = T.axis.reduce(n, i_t)
                            T.reads(
                                X_data[
                                    I_T_indices_data[vi_t + I_T_indptr_data[vj]] * feat_size + vf
                                ],
                                I_T_indices_data[vi_t + I_T_indptr_data[vj]],
                            )
                            T.writes(Y_data[vi * feat_size + vf])
                            T.block_attr({"sparse": True})
                            Y_data[vi * feat_size + vf] = (
                                Y_data[vi * feat_size + vf]
                                + X_data[
                                    I_T_indices_data[vi_t + I_T_indptr_data[vj]] * feat_size + vf
                                ]
                            )


@T.prim_func
def sddmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [m * feat_size], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [n * feat_size], dtype="float32", strides=[1])
    C_data = T.match_buffer(c, [nnz], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [m + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nnz], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for i in T.serial(m):
        with T.block("sddmm0"):
            vi = T.axis.spatial(m, i)
            T.reads(
                J_indptr_data[0 : m + 1],
                A_data[0 : m * feat_size],
                B_data[0 : n * feat_size],
                J_indices_data[0:nnz],
            )
            T.writes(C_data[0:nnz])
            T.block_attr({"sparse": True})
            for j, k in T.grid(J_indptr_data[vi + 1] - J_indptr_data[vi], feat_size):
                with T.block("sddmm1"):
                    vj = T.axis.spatial(n, j)
                    vk = T.axis.reduce(feat_size, k)
                    T.reads(
                        A_data[vi * feat_size + vk],
                        B_data[J_indices_data[vj + J_indptr_data[vi]] * feat_size + vk],
                        J_indices_data[vj + J_indptr_data[vi]],
                    )
                    T.writes(C_data[vj + J_indptr_data[vi]])
                    T.block_attr({"sparse": True})
                    with T.init():
                        C_data[vj + J_indptr_data[vi]] = T.float32(0)
                    C_data[vj + J_indptr_data[vi]] = (
                        C_data[vj + J_indptr_data[vi]]
                        + A_data[vi * feat_size + vk]
                        * B_data[J_indices_data[vj + J_indptr_data[vi]] * feat_size + vk]
                    )


@T.prim_func
def fused_sddmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [m * feat_size], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [n * feat_size], dtype="float32", strides=[1])
    C_data = T.match_buffer(c, [nnz], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [m + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nnz], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    mid_0_data = T.alloc_buffer([nnz], dtype="int32", strides=[1])
    for j in T.serial(nnz):
        with T.block("binary_search_block_0_0"):
            vi = T.axis.spatial(1, 0)
            vj = T.axis.spatial(nnz, j)
            T.reads()
            T.writes()
            T.block_attr({"preprocess": True, "sparse": True})
            low = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
            high = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
            low[0] = 0
            high[0] = m + 1
            while low[0] < high[0]:
                mid_0_data[vj] = low[0] + (high[0] - low[0]) // 2
                if J_indptr_data[mid_0_data[vj]] > vj:
                    high[0] = mid_0_data[vj]
                else:
                    low[0] = mid_0_data[vj] + 1
            mid_0_data[vj] = mid_0_data[vj] - 1
    for j, k in T.grid(nnz, feat_size):
        with T.block("sddmm0"):
            vi = T.axis.spatial(1, 0)
            vj, vk = T.axis.remap("SR", [j, k])
            T.reads(
                A_data[mid_0_data[vj + J_indptr_data[vi]] * feat_size + vk],
                mid_0_data[vj + J_indptr_data[vi]],
                B_data[J_indices_data[vj + J_indptr_data[vi]] * feat_size + vk],
                J_indices_data[vj + J_indptr_data[vi]],
            )
            T.writes(C_data[vj + J_indptr_data[vi]])
            T.block_attr({"sparse": True})
            with T.init():
                C_data[vj] = T.float32(0)
            C_data[vj] = (
                C_data[vj]
                + A_data[mid_0_data[vj] * feat_size + vk]
                * B_data[J_indices_data[vj] * feat_size + vk]
            )


@T.prim_func
def square_sum(
    a: T.handle,
    b: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    indptr_k: T.handle,
    indices_k: T.handle,
    nnz_j: T.int32,
    nnz_k: T.int32,
    M: T.int32,
    N1: T.int32,
    N2: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nnz_k], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [M], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr_j, [M + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices_j, [nnz_j], dtype="int32", strides=[1])
    K_indptr_data = T.match_buffer(indptr_k, [nnz_j + 1], dtype="int32", strides=[1])
    K_indices_data = T.match_buffer(indices_k, [nnz_k], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for i in T.serial(M):
        with T.block("square_sum0"):
            vi = T.axis.spatial(M, i)
            T.reads(J_indptr_data[0 : M + 1], K_indptr_data[0 : nnz_j + 1], A_data[0:nnz_k])
            T.writes(B_data[vi])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("square_sum1"):
                    vj = T.axis.reduce(N1, j)
                    T.reads(K_indptr_data[0 : nnz_j + 1], A_data[0:nnz_k])
                    T.writes(B_data[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B_data[vi] = T.float32(0)
                    for k in T.serial(
                        K_indptr_data[vj + J_indptr_data[vi] + 1]
                        - K_indptr_data[vj + J_indptr_data[vi]]
                    ):
                        with T.block("square_sum2"):
                            vk = T.axis.reduce(N2, k)
                            T.reads(A_data[vk + K_indptr_data[vj + J_indptr_data[vi]]])
                            T.writes(B_data[vi])
                            T.block_attr({"sparse": True})
                            B_data[vi] = (
                                B_data[vi] + A_data[vk + K_indptr_data[vj + J_indptr_data[vi]]]
                            )


@T.prim_func
def square_sum_two_K(
    a: T.handle,
    b: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    indptr_k0: T.handle,
    indices_k0: T.handle,
    indptr_k1: T.handle,
    indices_k1: T.handle,
    nnz_j: T.int32,
    nnz_k: T.int32,
    M: T.int32,
    N1: T.int32,
    N2: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nnz_k], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [M], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr_j, [M + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices_j, [nnz_j], dtype="int32", strides=[1])
    K0_indptr_data = T.match_buffer(indptr_k0, [nnz_j + 1], dtype="int32", strides=[1])
    K0_indices_data = T.match_buffer(indices_k0, [nnz_k], dtype="int32", strides=[1])
    K1_indptr_data = T.match_buffer(indptr_k1, [nnz_j + 1], dtype="int32", strides=[1])
    K1_indices_data = T.match_buffer(indices_k1, [nnz_k], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    mid_0_data = T.alloc_buffer([nnz_k], dtype="int32", strides=[1])
    for i in T.serial(M):
        with T.block("binary_search_block_0_0"):
            vi = T.axis.spatial(M, i)
            T.reads()
            T.writes()
            T.block_attr({"preprocess": True, "sparse": True})
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("binary_search_block_0_1"):
                    vj = T.axis.reduce(N1, j)
                    T.reads()
                    T.writes()
                    T.block_attr({"preprocess": True, "sparse": True})
                    for k in T.serial(
                        K1_indptr_data[vj + J_indptr_data[vi] + 1]
                        - K1_indptr_data[vj + J_indptr_data[vi]]
                    ):
                        with T.block("binary_search_block_0_2"):
                            vk = T.axis.reduce(N2, k)
                            T.reads()
                            T.writes()
                            T.block_attr({"preprocess": True, "sparse": True})
                            low = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
                            high = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
                            low[0] = 0
                            high[0] = (
                                K0_indptr_data[vj + J_indptr_data[vi] + 1]
                                - K0_indptr_data[vj + J_indptr_data[vi]]
                            )
                            while low[0] < high[0]:
                                mid_0_data[vk + K1_indptr_data[vj + J_indptr_data[vi]]] = (
                                    low[0] + (high[0] - low[0]) // 2
                                )
                                if (
                                    K0_indices_data[
                                        mid_0_data[vk + K1_indptr_data[vj + J_indptr_data[vi]]]
                                        + K0_indptr_data[vj + J_indptr_data[vi]]
                                    ]
                                    < K1_indices_data[vk + K1_indptr_data[vj + J_indptr_data[vi]]]
                                ):
                                    low[0] = (
                                        mid_0_data[vk + K1_indptr_data[vj + J_indptr_data[vi]]] + 1
                                    )
                                else:
                                    high[0] = mid_0_data[
                                        vk + K1_indptr_data[vj + J_indptr_data[vi]]
                                    ]
    for i in T.serial(M):
        with T.block("square_sum0"):
            vi_1 = T.axis.spatial(M, i)
            T.reads(
                J_indptr_data[0 : M + 1],
                K1_indptr_data[0 : nnz_j + 1],
                A_data[0:nnz_k],
                mid_0_data[0:nnz_k],
            )
            T.writes(B_data[vi_1])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr_data[vi_1 + 1] - J_indptr_data[vi_1]):
                with T.block("square_sum1"):
                    vj_1 = T.axis.reduce(N1, j)
                    T.reads(K1_indptr_data[0 : nnz_j + 1], A_data[0:nnz_k], mid_0_data[0:nnz_k])
                    T.writes(B_data[vi_1])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B_data[vi_1] = T.float32(0)
                    for k in T.serial(
                        K1_indptr_data[vj_1 + J_indptr_data[vi_1] + 1]
                        - K1_indptr_data[vj_1 + J_indptr_data[vi_1]]
                    ):
                        with T.block("square_sum2"):
                            vk = T.axis.reduce(N2, k)
                            T.reads(
                                A_data[
                                    mid_0_data[vk + K1_indptr_data[vj_1 + J_indptr_data[vi_1]]]
                                    + K0_indptr_data[vj_1 + J_indptr_data[vi_1]]
                                ],
                                mid_0_data[vk + K1_indptr_data[vj_1 + J_indptr_data[vi_1]]],
                            )
                            T.writes(B_data[vi_1])
                            T.block_attr({"sparse": True})
                            B_data[vi_1] = (
                                B_data[vi_1]
                                + A_data[
                                    mid_0_data[vk + K1_indptr_data[vj_1 + J_indptr_data[vi_1]]]
                                    + K0_indptr_data[vj_1 + J_indptr_data[vi_1]]
                                ]
                            )


@T.prim_func
def fused_reduction_4d_2d(
    x: T.handle,
    y: T.handle,
    indptr_j: T.handle,
    indptr_k: T.handle,
    indptr_l: T.handle,
    n: T.int32,
    nnz_j: T.int32,
    nnz_k: T.int32,
    nnz_l: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    X_data = T.match_buffer(x, [nnz_l], dtype="float32", strides=[1])
    Y_data = T.match_buffer(y, [nnz_j], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr_j, [n + 1], dtype="int32", strides=[1])
    K_indptr_data = T.match_buffer(indptr_k, [nnz_j + 1], dtype="int32", strides=[1])
    L_indptr_data = T.match_buffer(indptr_l, [nnz_k + 1], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for j in T.serial(nnz_j):
        with T.block("reduction_4d_2d0"):
            vi = T.axis.spatial(1, 0)
            vj = T.axis.spatial(nnz_j, j)
            T.reads(K_indptr_data[0 : nnz_j + 1], L_indptr_data[0 : nnz_k + 1], X_data[0:nnz_l])
            T.writes(Y_data[vj + J_indptr_data[vi]])
            T.block_attr({"sparse": True})
            for k in T.serial(K_indptr_data[vj + 1] - K_indptr_data[vj]):
                with T.block("reduction_4d_2d1"):
                    vk = T.axis.reduce(32768, k)
                    T.reads(L_indptr_data[0 : nnz_k + 1], X_data[0:nnz_l])
                    T.writes(Y_data[vj])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Y_data[vj] = T.float32(0)
                    for l in T.serial(
                        L_indptr_data[vk + K_indptr_data[vj] + 1]
                        - L_indptr_data[vk + K_indptr_data[vj]]
                    ):
                        with T.block("reduction_4d_2d2"):
                            vl = T.axis.reduce(32768, l)
                            T.reads(X_data[vl + L_indptr_data[vk + K_indptr_data[vj]]])
                            T.writes(Y_data[vj])
                            T.block_attr({"sparse": True})
                            Y_data[vj] = (
                                Y_data[vj] + X_data[vl + L_indptr_data[vk + K_indptr_data[vj]]]
                            )


@T.prim_func
def fused_reduction_4d_3d(
    x: T.handle,
    y: T.handle,
    indptr_j: T.handle,
    indptr_k: T.handle,
    indptr_l: T.handle,
    n: T.int32,
    nnz_j: T.int32,
    nnz_k: T.int32,
    nnz_l: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    X_data = T.match_buffer(x, [nnz_l], dtype="float32", strides=[1])
    Y_data = T.match_buffer(y, [nnz_k], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr_j, [n + 1], dtype="int32", strides=[1])
    K_indptr_data = T.match_buffer(indptr_k, [nnz_j + 1], dtype="int32", strides=[1])
    L_indptr_data = T.match_buffer(indptr_l, [nnz_k + 1], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for k in T.serial(nnz_k):
        with T.block("reduction_4d_3d0"):
            vi = T.axis.spatial(1, 0)
            vj = T.axis.spatial(1, 0)
            vk = T.axis.spatial(nnz_k, k)
            T.reads(L_indptr_data[0 : nnz_k + 1], X_data[0:nnz_l])
            T.writes(Y_data[vk + K_indptr_data[vj + J_indptr_data[vi]]])
            T.block_attr({"sparse": True})
            for l in T.serial(L_indptr_data[vk + 1] - L_indptr_data[vk]):
                with T.block("reduction_4d_3d1"):
                    vl = T.axis.reduce(32768, l)
                    T.reads(X_data[vl + L_indptr_data[vk]])
                    T.writes(Y_data[vk])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Y_data[vk] = T.float32(0)
                    Y_data[vk] = Y_data[vk] + X_data[vl + L_indptr_data[vk]]


@T.prim_func
def rgcn_forward(
    etype: T.handle,
    w: T.handle,
    x: T.handle,
    y: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    r: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    E_data = T.match_buffer(etype, [nnz], dtype="int32", strides=[1])
    W_data = T.match_buffer(w, [r * feat_size * feat_size], dtype="float32", strides=[1])
    X_data = T.match_buffer(x, [n * feat_size], dtype="float32", strides=[1])
    Y_data = T.match_buffer(y, [n * feat_size], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [n + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nnz], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for i, fo in T.grid(n, feat_size):
        with T.block("rgcn-forward0"):
            vi, vfo = T.axis.remap("SS", [i, fo])
            T.reads(
                J_indptr_data[0 : n + 1],
                W_data[0 : r * feat_size * feat_size],
                E_data[0:nnz],
                X_data[0 : n * feat_size],
                J_indices_data[0:nnz],
            )
            T.writes(Y_data[vi * feat_size + vfo])
            T.block_attr({"sparse": True})
            for j, fi in T.grid(J_indptr_data[vi + 1] - J_indptr_data[vi], feat_size):
                with T.block("rgcn-forward1"):
                    vj = T.axis.reduce(n, j)
                    vfi = T.axis.reduce(feat_size, fi)
                    T.reads(
                        W_data[
                            E_data[vj + J_indptr_data[vi]] * (feat_size * feat_size)
                            + vfo * feat_size
                            + vfi
                        ],
                        E_data[vj + J_indptr_data[vi]],
                        X_data[J_indices_data[vj + J_indptr_data[vi]] * feat_size + vfi],
                        J_indices_data[vj + J_indptr_data[vi]],
                    )
                    T.writes(Y_data[vi * feat_size + vfo])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Y_data[vi * feat_size + vfo] = T.float32(0)
                    Y_data[vi * feat_size + vfo] = (
                        Y_data[vi * feat_size + vfo]
                        + W_data[
                            E_data[vj + J_indptr_data[vi]] * (feat_size * feat_size)
                            + vfo * feat_size
                            + vfi
                        ]
                        * X_data[J_indices_data[vj + J_indptr_data[vi]] * feat_size + vfi]
                    )


@T.prim_func
def rgcn_hetero_forward(
    w: T.handle,
    x: T.handle,
    y: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    n: T.int32,
    num_rels: T.int32,
    feat_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    W_data = T.match_buffer(w, [num_rels * feat_size * feat_size], dtype="float32", strides=[1])
    X_data = T.match_buffer(x, [n * feat_size], dtype="float32", strides=[1])
    Y_data = T.match_buffer(y, [n * feat_size], dtype="float32", strides=[1])
    I_indptr_data = T.match_buffer(indptr_i, [num_rels + 1], dtype="int32", strides=[1])
    I_indices_data = T.match_buffer(indices_i, [nnz_i], dtype="int32", strides=[1])
    J_indptr_data = T.match_buffer(indptr_j, [nnz_i + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices_j, [nnz_j], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    for fo, r in T.grid(feat_size, num_rels):
        with T.block("rgcn-hetero-forward0"):
            vfo, vr = T.axis.remap("SS", [fo, r])
            T.reads(
                I_indptr_data[0 : num_rels + 1],
                J_indptr_data[0 : nnz_i + 1],
                I_indices_data[0:nnz_i],
                W_data[0 : num_rels * feat_size * feat_size],
                X_data[0 : n * feat_size],
                J_indices_data[0:nnz_j],
            )
            T.writes(Y_data[0 : n * feat_size])
            T.block_attr({"sparse": True})
            for i in T.serial(I_indptr_data[vr + 1] - I_indptr_data[vr]):
                with T.block("rgcn-hetero-forward1"):
                    vi = T.axis.spatial(n, i)
                    T.reads(
                        J_indptr_data[0 : nnz_i + 1],
                        I_indices_data[vi + I_indptr_data[vr]],
                        W_data[0 : num_rels * feat_size * feat_size],
                        X_data[0 : n * feat_size],
                        J_indices_data[0:nnz_j],
                    )
                    T.writes(Y_data[I_indices_data[vi + I_indptr_data[vr]] * feat_size + vfo])
                    T.block_attr({"sparse": True})
                    for j, fi in T.grid(
                        J_indptr_data[vi + I_indptr_data[vr] + 1]
                        - J_indptr_data[vi + I_indptr_data[vr]],
                        feat_size,
                    ):
                        with T.block("rgcn-hetero-forward2"):
                            vj = T.axis.reduce(n, j)
                            vfi = T.axis.reduce(feat_size, fi)
                            T.reads(
                                I_indices_data[vi + I_indptr_data[vr]],
                                W_data[vr * (feat_size * feat_size) + vfo * feat_size + vfi],
                                X_data[
                                    J_indices_data[vj + J_indptr_data[vi + I_indptr_data[vr]]]
                                    * feat_size
                                    + vfi
                                ],
                                J_indices_data[vj + J_indptr_data[vi + I_indptr_data[vr]]],
                            )
                            T.writes(
                                Y_data[I_indices_data[vi + I_indptr_data[vr]] * feat_size + vfo]
                            )
                            T.block_attr({"sparse": True})
                            with T.init():
                                Y_data[
                                    I_indices_data[vi + I_indptr_data[vr]] * feat_size + vfo
                                ] = T.float32(0)
                            Y_data[I_indices_data[vi + I_indptr_data[vr]] * feat_size + vfo] = (
                                Y_data[I_indices_data[vi + I_indptr_data[vr]] * feat_size + vfo]
                                + W_data[vr * (feat_size * feat_size) + vfo * feat_size + vfi]
                                * X_data[
                                    J_indices_data[vj + J_indptr_data[vi + I_indptr_data[vr]]]
                                    * feat_size
                                    + vfi
                                ]
                            )


@T.prim_func
def sparse_softmax(
    a: T.handle, b: T.handle, indptr: T.handle, indices: T.handle, n: T.int32, nnz: T.int32
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nnz], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [nnz], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr, [n + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices, [nnz], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    TMP_data = T.alloc_buffer([n], dtype="float32", strides=[1])
    TMP1_data = T.alloc_buffer([n], dtype="float32", strides=[1])
    for i in T.serial(n):
        with T.block("sparse_softmax0"):
            vi = T.axis.spatial(n, i)
            T.reads(J_indptr_data[0 : n + 1], A_data[0:nnz], TMP_data[vi], TMP1_data[vi])
            T.writes(TMP_data[vi], TMP1_data[vi], B_data[0:nnz])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("computer_max0"):
                    vj = T.axis.reduce(n, j)
                    T.reads(A_data[vj + J_indptr_data[vi]])
                    T.writes(TMP_data[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        TMP_data[vi] = T.float32(-100000)
                    TMP_data[vi] = T.max(TMP_data[vi], A_data[vj + J_indptr_data[vi]])
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("exp_and_sum0"):
                    vj = T.axis.reduce(n, j)
                    T.reads(A_data[vj + J_indptr_data[vi]], TMP_data[vi])
                    T.writes(TMP1_data[vi])
                    T.block_attr({"sparse": True})
                    with T.init():
                        TMP1_data[vi] = T.float32(-100000)
                    TMP1_data[vi] = TMP1_data[vi] + T.exp(
                        A_data[vj + J_indptr_data[vi]] - TMP_data[vi], dtype="float32"
                    )
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("div0"):
                    vj = T.axis.spatial(n, j)
                    T.reads(A_data[vj + J_indptr_data[vi]], TMP1_data[vi])
                    T.writes(B_data[vj + J_indptr_data[vi]])
                    T.block_attr({"sparse": True})
                    B_data[vj + J_indptr_data[vi]] = (
                        T.exp(A_data[vj + J_indptr_data[vi]], dtype="float32") / TMP1_data[vi]
                    )


@T.prim_func
def csr2bsr(
    a: T.handle,
    b: T.handle,
    indptr_in: T.handle,
    indices_in: T.handle,
    indptr_out: T.handle,
    indices_out: T.handle,
    m_in: T.int32,
    n_in: T.int32,
    m_out: T.int32,
    n_out: T.int32,
    nnz_in: T.int32,
    nnz_out: T.int32,
    blk_size: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 0})
    A_data = T.match_buffer(a, [nnz_in], dtype="float32", strides=[1])
    B_data = T.match_buffer(b, [nnz_out * blk_size * blk_size], dtype="float32", strides=[1])
    J_indptr_data = T.match_buffer(indptr_in, [m_in + 1], dtype="int32", strides=[1])
    J_indices_data = T.match_buffer(indices_in, [nnz_in], dtype="int32", strides=[1])
    J_bsr_indptr_data = T.match_buffer(indptr_out, [m_out + 1], dtype="int32", strides=[1])
    J_bsr_indices_data = T.match_buffer(indices_out, [nnz_out], dtype="int32", strides=[1])
    # body
    # with T.block("root")
    mid_0_data = T.alloc_buffer([nnz_in], dtype="int32", strides=[1])
    for i in T.serial(m_in):
        with T.block("binary_search_block_0_0"):
            vi = T.axis.spatial(m_in, i)
            T.reads()
            T.writes()
            T.block_attr({"preprocess": True, "sparse": True})
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("binary_search_block_0_1"):
                    vj = T.axis.spatial(n_in, j)
                    T.reads()
                    T.writes()
                    T.block_attr({"preprocess": True, "sparse": True})
                    low = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
                    high = T.alloc_buffer([1], dtype="int32", strides=[1], scope="local")
                    low[0] = 0
                    high[0] = (
                        J_bsr_indptr_data[vi // blk_size + 1] - J_bsr_indptr_data[vi // blk_size]
                    )
                    while low[0] < high[0]:
                        mid_0_data[vj + J_indptr_data[vi]] = low[0] + (high[0] - low[0]) // 2
                        if (
                            J_bsr_indices_data[
                                mid_0_data[vj + J_indptr_data[vi]]
                                + J_bsr_indptr_data[vi // blk_size]
                            ]
                            < J_indices_data[vj + J_indptr_data[vi]] // blk_size
                        ):
                            low[0] = mid_0_data[vj + J_indptr_data[vi]] + 1
                        else:
                            high[0] = mid_0_data[vj + J_indptr_data[vi]]
    for i in T.serial(m_in):
        with T.block("csr2bsr0"):
            vi = T.axis.spatial(m_in, i)
            T.reads(
                J_indptr_data[0 : m_in + 1],
                A_data[0:nnz_in],
                mid_0_data[0:nnz_in],
                J_indices_data[0:nnz_in],
            )
            T.writes(B_data[0 : nnz_out * blk_size * blk_size])
            T.block_attr({"sparse": True})
            for j in T.serial(J_indptr_data[vi + 1] - J_indptr_data[vi]):
                with T.block("csr2bsr1"):
                    vj = T.axis.spatial(n_in, j)
                    T.reads(
                        A_data[vj + J_indptr_data[vi]],
                        mid_0_data[vj + J_indptr_data[vi]],
                        J_indices_data[vj + J_indptr_data[vi]],
                    )
                    T.writes(B_data[0 : nnz_out * blk_size * blk_size])
                    T.block_attr({"sparse": True})
                    B_data[
                        (mid_0_data[vj + J_indptr_data[vi]] + J_bsr_indptr_data[vi // blk_size])
                        * (blk_size * blk_size)
                        + vi % blk_size * blk_size
                        + J_indices_data[vj + J_indptr_data[vi]] % blk_size
                    ] = A_data[vj + J_indptr_data[vi]]
