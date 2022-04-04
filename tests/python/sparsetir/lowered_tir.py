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

"""Lowered TIR scripts of sparse workloads."""
from tvm.script import tir as T


@T.prim_func
def lowered_csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (nnz,), "float32")
    B_data = T.match_buffer(b, (n * k,), "float32")
    C_data = T.match_buffer(c, (m * k,), "float32")
    J_indptr = T.match_buffer(indptr, (m + 1,), "int32")
    J_indices = T.match_buffer(indices, (nnz,), "int32")
    # body
    # with T.block("root")
    for v_vi, v_vk in T.grid(m, k):
        with T.block("csrmm0"):
            vi, vk = T.axis.remap("SS", [v_vi, v_vk])
            T.reads(
                J_indptr[0 : m + 1],
                J_indices[0:nnz],
                A_data[0:nnz],
                B_data[0 : n * k],
                C_data[0 : m * k],
            )
            T.writes(C_data[0 : m * k])
            T.block_attr({"sparse": True})
            for v_vj in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("csrmm1"):
                    vj = T.axis.reduce(J_indptr[vi + 1] - J_indptr[vi], v_vj)
                    T.reads(
                        J_indptr[0 : m + 1],
                        J_indices[0:nnz],
                        A_data[0:nnz],
                        B_data[0 : n * k],
                        C_data[0 : m * k],
                    )
                    T.writes(C_data[0 : m * k])
                    T.block_attr({"sparse": True})
                    with T.init():
                        C_data[vi * k + vk] = T.float32(0)
                    C_data[vi * k + vk] = (
                        C_data[vi * k + vk]
                        + A_data[J_indptr[vi] + vj] * B_data[J_indices[J_indptr[vi] + vj] * k + vk]
                    )


# @T.prim_func
# def lowered_csrmm_dense_iter(
#     a: T.handle,
#     b: T.handle,
#     c: T.handle,
#     indptr: T.handle,
#     indices: T.handle,
#     m: T.int32,
#     n: T.int32,
#     k: T.int32,
#     nnz: T.int32,
# ) -> None:
#     # function attr dict
#     T.func_attr({"global_symbol": "main", "tir.noalias": True})
#     A_data = T.match_buffer(a, (nnz,), "float32")
#     B_data = T.match_buffer(b, (n * k,), "float32")
#     C_data = T.match_buffer(c, (m * k,), "float32")
#     J_indptr = T.match_buffer(indptr, (m + 1,), "int32")
#     J_indices = T.match_buffer(indices, (nnz,), "int32")
#     # body
#     # with T.block("root")
#     for v_vi, v_vj, v_vk in T.grid(m, n, k):
#         with T.block("csrmm0"):
#             vi, vj, vk = T.axis.remap("SRS", [v_vi, v_vj, v_vk])
#             T.reads(
#                 J_indptr[0 : m + 1],
#                 J_indices[0:nnz],
#                 A_data[0:nnz],
#                 B_data[0 : n * k],
#                 C_data[0 : m * k],
#             )
#             T.writes(C_data[0 : m * k])
#             T.block_attr({"sparse": True})
#             with T.init():
#                 C_data[vi * k + vk] = T.float32(0)
#             C_data[vi * k + vk] = (
#                 C_data[vi * k + vk]
#                 + A_data[
#                     T.tvm_lower_bound(
#                         J_indices.data, vj, J_indptr[vi], J_indptr[vi + 1], dtype="int32"
#                     )
#                 ]
#                 * B_data[vj * k + vk]
#             )


@T.prim_func
def lowered_csr_reduce(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, [nnz], dtype="float32")
    B_data = T.match_buffer(b, [n], dtype="float32")
    J_indptr = T.match_buffer(indptr, [n + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnz], dtype="int32")
    for v_vi in T.serial(0, n):
        with T.block("csr_reduce_outer"):
            vi = T.axis.spatial(n, v_vi)
            T.reads([J_indptr[0 : n + 1], J_indices[0:nnz], A_data[0:nnz], B_data[0:n]])
            T.writes([B_data[0:n]])
            T.block_attr({"sparse": True})
            for v_vj in T.serial(0, J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("csr_reduce"):
                    vj = T.axis.reduce(J_indptr[vi + 1] - J_indptr[vi], v_vj)
                    T.reads([J_indptr[0 : n + 1], J_indices[0:nnz], A_data[0:nnz], B_data[0:n]])
                    T.writes([B_data[0:n]])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B_data[vi] = T.float32(0)
                    B_data[vi] = B_data[vi] + A_data[J_indptr[vi] + vj]


@T.prim_func
def lowered_segment_reduce(
    a: T.handle, b: T.handle, indptr: T.handle, n: T.int32, nnz: T.int32
) -> None:
    A_data = T.match_buffer(a, (nnz,), "float32")
    B_data = T.match_buffer(b, (n,), "float32")
    J_indptr = T.match_buffer(indptr, (n + 1,), "int32")
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    for v_vi in T.serial(n):
        with T.block("segment_reduce0"):
            vi = T.axis.spatial(n, v_vi)
            T.reads(J_indptr[0 : n + 1], A_data[0:nnz], B_data[0:n])
            T.writes(B_data[0:n])
            T.block_attr({"sparse": True})
            for v_vj in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("segment_reduce1"):
                    vj = T.axis.reduce(J_indptr[vi + 1] - J_indptr[vi], v_vj)
                    T.reads(J_indptr[0 : n + 1], A_data[0:nnz], B_data[0:n])
                    T.writes(B_data[0:n])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B_data[vi] = T.float32(0)
                    B_data[vi] = B_data[vi] + A_data[J_indptr[vi] + vj]


@T.prim_func
def lowered_bsrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    j_indptr: T.handle,
    j_indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    nnzb: T.int32,
    blk: T.int32,
    feat_size: T.int32,
) -> None:
    A_data = T.match_buffer(a, (nnzb * blk * blk,), "float32")
    B_data = T.match_buffer(b, (mb * blk * feat_size,), "float32")
    C_data = T.match_buffer(c, (nb * blk * feat_size,), "float32")
    J_indptr = T.match_buffer(j_indptr, (nb + 1,), "int32")
    J_indices = T.match_buffer(j_indices, (nnzb,), "int32")
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    for v_vi, v_vbi, v_vbj, v_vf in T.grid(nb, blk, blk, feat_size):
        with T.block("bsrmm0"):
            vi, vbi, vbj, vf = T.axis.remap("SSRS", [v_vi, v_vbi, v_vbj, v_vf])
            T.reads(
                J_indptr[0 : nb + 1],
                J_indices[0:nnzb],
                A_data[0 : nnzb * blk * blk],
                B_data[0 : mb * blk * feat_size],
                C_data[0 : nb * blk * feat_size],
            )
            T.writes(C_data[0 : nb * blk * feat_size])
            T.block_attr({"sparse": True})
            with T.init():
                C_data[(vi * blk + vbi) * feat_size + vf] = T.float32(0)
            for v_vj in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("bsrmm1"):
                    vj = T.axis.reduce(J_indptr[vi + 1] - J_indptr[vi], v_vj)
                    T.reads(
                        J_indptr[0 : nb + 1],
                        J_indices[0:nnzb],
                        A_data[0 : nnzb * blk * blk],
                        B_data[0 : mb * blk * feat_size],
                        C_data[0 : nb * blk * feat_size],
                    )
                    T.writes(C_data[0 : nb * blk * feat_size])
                    T.block_attr({"sparse": True})
                    C_data[(vi * blk + vbi) * feat_size + vf] = (
                        C_data[(vi * blk + vbi) * feat_size + vf]
                        + A_data[((J_indptr[vi] + vj) * blk + vbi) * blk + vbj]
                        * B_data[(J_indices[J_indptr[vi] + vj] * blk + vbj) * feat_size + vf]
                    )


@T.prim_func
def lowered_ellmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    j_indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    feat_size: T.int32,
    col: T.int32,
    blk: T.int32,
) -> None:
    A_data = T.match_buffer(a, (nb * col * blk * blk,), "float32")
    B_data = T.match_buffer(b, (mb * blk * feat_size,), "float32")
    C_data = T.match_buffer(c, (nb * blk * feat_size,), "float32")
    J_indices = T.match_buffer(j_indices, (nb * col,), "int32")
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    for v_vi, v_vj, v_vbi, v_vbj, v_vf in T.grid(nb, col, blk, blk, feat_size):
        with T.block("ellmm0"):
            vi, vj, vbi, vbj, vf = T.axis.remap("SRSRS", [v_vi, v_vj, v_vbi, v_vbj, v_vf])
            T.reads(
                J_indices[0 : nb * col],
                A_data[0 : nb * col * blk * blk],
                B_data[0 : mb * blk * feat_size],
                C_data[0 : nb * blk * feat_size],
            )
            T.writes(C_data[0 : nb * blk * feat_size])
            T.block_attr({"sparse": True})
            with T.init():
                C_data[(vi * blk + vbi) * feat_size + vf] = T.float32(0)
            C_data[(vi * blk + vbi) * feat_size + vf] = (
                C_data[(vi * blk + vbi) * feat_size + vf]
                + A_data[((vi * col + vj) * blk + vbi) * blk + vbj]
                * B_data[(J_indices[vi * col + vj] * blk + vbj) * feat_size + vf]
            )


@T.prim_func
def lowered_sddmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (m * k,), "float32")
    B_data = T.match_buffer(b, (n * k,), "float32")
    C_data = T.match_buffer(c, (nnz,), "float32")
    J_indptr = T.match_buffer(indptr, (m + 1,), "int32")
    J_indices = T.match_buffer(indices, (nnz,), "int32")
    for v_vi in T.serial(m):
        with T.block("sddmm0"):
            vi = T.axis.spatial(m, v_vi)
            T.reads(
                J_indptr[0 : m + 1],
                J_indices[0:nnz],
                A_data[0 : m * k],
                B_data[0 : n * k],
                C_data[0:nnz],
            )
            T.writes(C_data[0:nnz])
            T.block_attr({"sparse": True})
            for v_vj, v_vk in T.grid(J_indptr[vi + 1] - J_indptr[vi], k):
                with T.block("sddmm1"):
                    vj, vk = T.axis.remap("SR", [v_vj, v_vk])
                    T.reads(
                        J_indptr[0 : m + 1],
                        J_indices[0:nnz],
                        A_data[0 : m * k],
                        B_data[0 : n * k],
                        C_data[0:nnz],
                    )
                    T.writes(C_data[0:nnz])
                    T.block_attr({"sparse": True})
                    with T.init():
                        C_data[J_indptr[vi] + vj] = T.float32(0)
                    C_data[J_indptr[vi] + vj] = (
                        C_data[J_indptr[vi] + vj]
                        + A_data[vi * k + vk] * B_data[J_indices[J_indptr[vi] + vj] * k + vk]
                    )


# from tvm.script import tir as T
@T.prim_func
def lowered_sddmm_fuse(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (m * k,), "float32")
    B_data = T.match_buffer(b, (n * k,), "float32")
    C_data = T.match_buffer(c, (nnz,), "float32")
    J_indptr = T.match_buffer(indptr, (m + 1,), "int32")
    J_indices = T.match_buffer(indices, (nnz,), "int32")
    # body
    # with T.block("root")
    for v_vi, v_vj, v_vk in T.grid(1, nnz, k):
        with T.block("sddmm0"):
            vi, vj, vk = T.axis.remap("SSR", [v_vi, v_vj, v_vk])
            T.reads(
                J_indptr[0 : m + 1],
                J_indices[0:nnz],
                A_data[0 : m * k],
                B_data[0 : n * k],
                C_data[0:nnz],
            )
            T.writes(C_data[0:nnz])
            T.block_attr({"sparse": True})
            with T.init():
                C_data[vj] = T.float32(0)
            C_data[vj] = (
                C_data[vj]
                + A_data[
                    (T.tvm_upper_bound(J_indptr.data, vj, 0, m + 1, dtype="int32") - 1) * k + vk
                ]
                * B_data[J_indices[vj] * k + vk]
            )


@T.prim_func
def lowered_bmm(
    x: T.handle,
    y: T.handle,
    z: T.handle,
    indptr_i: T.handle,
    indptr_j: T.handle,
    indptr_k: T.handle,
    indptr_ij: T.handle,
    indptr_jk: T.handle,
    indptr_ik: T.handle,
    batch_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
    nnz_k: T.int32,
    nnz_ij: T.int32,
    nnz_jk: T.int32,
    nnz_ik: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    X_data = T.match_buffer(x, (nnz_ij,), "float32")
    Y_data = T.match_buffer(y, (nnz_jk,), "float32")
    Z_data = T.match_buffer(z, (nnz_ik,), "float32")
    I_indptr = T.match_buffer(indptr_i, (batch_size + 1,), "int32")
    J_indptr = T.match_buffer(indptr_j, (batch_size + 1,), "int32")
    K_indptr = T.match_buffer(indptr_k, (batch_size + 1,), "int32")
    IJ_indptr = T.match_buffer(indptr_ij, (batch_size + 1,), "int32")
    JK_indptr = T.match_buffer(indptr_jk, (batch_size + 1,), "int32")
    IK_indptr = T.match_buffer(indptr_ik, (batch_size + 1,), "int32")
    # body
    # with T.block("root")
    for v_vb in T.serial(batch_size):
        with T.block("bmm0"):
            vb = T.axis.spatial(batch_size, v_vb)
            T.reads(
                I_indptr[0 : batch_size + 1],
                J_indptr[0 : batch_size + 1],
                K_indptr[0 : batch_size + 1],
                IJ_indptr[0 : batch_size + 1],
                JK_indptr[0 : batch_size + 1],
                IK_indptr[0 : batch_size + 1],
                X_data[0:nnz_ij],
                Y_data[0:nnz_jk],
                Z_data[0:nnz_ik],
            )
            T.writes(Z_data[0:nnz_ik])
            T.block_attr({"sparse": True})
            for v_vi, v_vj, v_vk in T.grid(
                I_indptr[vb + 1] - I_indptr[vb],
                J_indptr[vb + 1] - J_indptr[vb],
                K_indptr[vb + 1] - K_indptr[vb],
            ):
                with T.block("bmm1"):
                    vi, vj, vk = T.axis.remap("SRS", [v_vi, v_vj, v_vk])
                    T.reads(
                        I_indptr[0 : batch_size + 1],
                        J_indptr[0 : batch_size + 1],
                        K_indptr[0 : batch_size + 1],
                        IJ_indptr[0 : batch_size + 1],
                        JK_indptr[0 : batch_size + 1],
                        IK_indptr[0 : batch_size + 1],
                        X_data[0:nnz_ij],
                        Y_data[0:nnz_jk],
                        Z_data[0:nnz_ik],
                    )
                    T.writes(Z_data[0:nnz_ik])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Z_data[
                            IK_indptr[vb] + vi * (K_indptr[vb + 1] - K_indptr[vb]) + vk
                        ] = T.float32(0)
                    Z_data[IK_indptr[vb] + vi * (K_indptr[vb + 1] - K_indptr[vb]) + vk] = (
                        Z_data[IK_indptr[vb] + vi * (K_indptr[vb + 1] - K_indptr[vb]) + vk]
                        + X_data[IJ_indptr[vb] + vi * (J_indptr[vb + 1] - J_indptr[vb]) + vj]
                        * Y_data[JK_indptr[vb] + vj * (K_indptr[vb + 1] - K_indptr[vb]) + vk]
                    )


@T.prim_func
def lowered_square_sum(
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
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, [nnz_k], dtype="float32")
    B_data = T.match_buffer(b, [M], dtype="float32")
    J_indptr = T.match_buffer(indptr_j, [M + 1], dtype="int32")
    J_indices = T.match_buffer(indices_j, [nnz_j], dtype="int32")
    K_indptr = T.match_buffer(indptr_k, [nnz_j + 1], dtype="int32")
    K_indices = T.match_buffer(indices_k, [nnz_k], dtype="int32")

    for v_vi in T.serial(0, M):
        with T.block("square_sum_2"):
            vi = T.axis.spatial(M, v_vi)
            T.reads(
                [
                    J_indptr[0 : M + 1],
                    J_indices[0:nnz_j],
                    K_indptr[0 : nnz_j + 1],
                    K_indices[0:nnz_k],
                    A_data[0:nnz_k],
                    B_data[0:M],
                ]
            )
            T.writes([B_data[0:M]])
            T.block_attr({"sparse": True})
            for v_vj in T.serial(0, J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("square_sum_1"):
                    vj = T.axis.reduce(J_indptr[vi + 1] - J_indptr[vi], v_vj)
                    T.reads(
                        [
                            J_indptr[0 : M + 1],
                            J_indices[0:nnz_j],
                            K_indptr[0 : nnz_j + 1],
                            K_indices[0:nnz_k],
                            A_data[0:nnz_k],
                            B_data[0:M],
                        ]
                    )
                    T.writes([B_data[0:M]])
                    T.block_attr({"sparse": True})
                    with T.init():
                        B_data[vi] = T.float32(0)
                    for v_vk in T.serial(
                        0, K_indptr[J_indptr[vi] + vj + 1] - K_indptr[J_indptr[vi] + vj]
                    ):
                        with T.block("square_sum"):
                            vk = T.axis.reduce(
                                K_indptr[J_indptr[vi] + vj + 1] - K_indptr[J_indptr[vi] + vj], v_vk
                            )
                            T.reads(
                                [
                                    J_indptr[0 : M + 1],
                                    J_indices[0:nnz_j],
                                    K_indptr[0 : nnz_j + 1],
                                    K_indices[0:nnz_k],
                                    A_data[0:nnz_k],
                                    B_data[0:M],
                                ]
                            )
                            T.writes([B_data[0:M]])
                            T.block_attr({"sparse": True})
                            B_data[vi] = B_data[vi] + A_data[K_indptr[J_indptr[vi] + vj] + vk]


# @T.prim_func
# def lowered_square_sum_two_K(
#     a: T.handle,
#     b: T.handle,
#     indptr_j: T.handle,
#     indices_j: T.handle,
#     indptr_k0: T.handle,
#     indices_k0: T.handle,
#     indptr_k1: T.handle,
#     indices_k1: T.handle,
#     nnz_j: T.int32,
#     nnz_k: T.int32,
#     M: T.int32,
#     N1: T.int32,
#     N2: T.int32,
# ) -> None:
#     T.func_attr({"global_symbol": "main", "tir.noalias": True})
#     A_data = T.match_buffer(a, [nnz_k], dtype="float32")
#     B_data = T.match_buffer(b, [M], dtype="float32")
#     J_indptr = T.match_buffer(indptr_j, [M + 1], dtype="int32")
#     J_indices = T.match_buffer(indices_j, [nnz_j], dtype="int32")
#     K0_indptr = T.match_buffer(indptr_k0, [nnz_j + 1], dtype="int32")
#     K0_indices = T.match_buffer(indices_k0, [nnz_k], dtype="int32")
#     K1_indptr = T.match_buffer(indptr_k1, [nnz_j + 1], dtype="int32")
#     K1_indices = T.match_buffer(indices_k1, [nnz_k], dtype="int32")

#     for v_vi in T.serial(0, M):
#         with T.block("square_sum_2"):
#             vi = T.axis.spatial(M, v_vi)
#             T.reads(
#                 [
#                     J_indptr[0 : M + 1],
#                     J_indices[0:nnz_j],
#                     K0_indptr[0 : nnz_j + 1],
#                     K0_indices[0:nnz_k],
#                     K1_indptr[0 : nnz_j + 1],
#                     K1_indices[0:nnz_k],
#                     A_data[0:nnz_k],
#                     B_data[0:M],
#                 ]
#             )
#             T.writes([B_data[0:M]])
#             T.block_attr({"sparse": True})
#             for v_vj in T.serial(0, J_indptr[vi + 1] - J_indptr[vi]):
#                 with T.block("square_sum_1"):
#                     vj = T.axis.reduce(J_indptr[vi + 1] - J_indptr[vi], v_vj)
#                     T.reads(
#                         [
#                             J_indptr[0 : M + 1],
#                             J_indices[0:nnz_j],
#                             K0_indptr[0 : nnz_j + 1],
#                             K0_indices[0:nnz_k],
#                             K1_indptr[0 : nnz_j + 1],
#                             K1_indices[0:nnz_k],
#                             A_data[0:nnz_k],
#                             B_data[0:M],
#                         ]
#                     )
#                     T.writes([B_data[0:M]])
#                     T.block_attr({"sparse": True})
#                     with T.init():
#                         B_data[vi] = T.float32(0)
#                     for v_vk in T.serial(
#                         0, K1_indptr[J_indptr[vi] + vj + 1] - K1_indptr[J_indptr[vi] + vj]
#                     ):
#                         with T.block("square_sum"):
#                             vk = T.axis.reduce(
#                                 K1_indptr[J_indptr[vi] + vj + 1] - K1_indptr[J_indptr[vi] + vj],
#                                 v_vk,
#                             )
#                             T.reads(
#                                 [
#                                     J_indptr[0 : M + 1],
#                                     J_indices[0:nnz_j],
#                                     K0_indptr[0 : nnz_j + 1],
#                                     K0_indices[0:nnz_k],
#                                     K1_indptr[0 : nnz_j + 1],
#                                     K1_indices[0:nnz_k],
#                                     A_data[0:nnz_k],
#                                     B_data[0:M],
#                                 ]
#                             )
#                             T.writes([B_data[0:M]])
#                             T.block_attr({"sparse": True})
#                             B_data[vi] = (
#                                 B_data[vi]
#                                 + A_data[
#                                     T.tvm_lower_bound(
#                                         K0_indices.data,
#                                         K1_indices[K1_indptr[J_indptr[vi] + vj] + vk],
#                                         K0_indptr[J_indptr[vi] + vj],
#                                         K0_indptr[J_indptr[vi] + vj + 1],
#                                         dtype="int32",
#                                     )
#                                 ]
#                             )


@T.prim_func
def lowered_csr_element_wise(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, [nnz], dtype="float32")
    B_data = T.match_buffer(b, [nnz], dtype="float32")
    J_indptr = T.match_buffer(indptr, [m + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnz], dtype="int32")
    for v_vi in T.serial(0, m):
        with T.block("csr_element_wise_outer"):
            vi = T.axis.spatial(m, v_vi)
            T.reads([J_indptr[0 : m + 1], J_indices[0:nnz], A_data[0:nnz]])
            T.writes([B_data[0:nnz]])
            T.block_attr({"sparse": True})
            for v_vj in T.serial(0, J_indptr[vi + 1] - J_indptr[vi]):
                with T.block("csr_element_wise"):
                    vj = T.axis.spatial(J_indptr[vi + 1] - J_indptr[vi], v_vj)
                    T.reads([J_indptr[0 : m + 1], J_indices[0:nnz], A_data[0:nnz]])
                    T.writes([B_data[0:nnz]])
                    T.block_attr({"sparse": True})
                    B_data[J_indptr[vi] + vj] = A_data[J_indptr[vi] + vj] * T.float32(2.5)


@T.prim_func
def lowered_rgcn_forward(
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
    E_data = T.match_buffer(etype, [nnz], dtype="int32")
    W_data = T.match_buffer(w, [r * feat_size * feat_size], dtype="float32")
    X_data = T.match_buffer(x, [n * feat_size], dtype="float32")
    Y_data = T.match_buffer(y, [n * feat_size], dtype="float32")
    J_indptr = T.match_buffer(indptr, [n + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnz], dtype="int32")
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    for v_vi, v_vout in T.grid(n, feat_size):
        with T.block("rgcn-forward_0"):
            vi, vout = T.axis.remap("SS", [v_vi, v_vout])
            T.reads(
                J_indptr[0 : n + 1],
                J_indices[0:nnz],
                E_data[0:nnz],
                W_data[0 : r * feat_size * feat_size],
                X_data[0 : n * feat_size],
                Y_data[0 : n * feat_size],
            )
            T.writes(Y_data[0 : n * feat_size])
            T.block_attr({"sparse": True})
            for v_vj in T.serial(J_indptr[vi + 1] - J_indptr[vi]):
                for v_vin in T.serial(feat_size):
                    with T.block("rgcn-forward_1"):
                        vj, vin = T.axis.remap("RR", [v_vj, v_vin])
                        T.reads(
                            J_indptr[0 : n + 1],
                            J_indices[0:nnz],
                            E_data[0:nnz],
                            W_data[0 : r * feat_size * feat_size],
                            X_data[0 : n * feat_size],
                            Y_data[0 : n * feat_size],
                        )
                        T.writes(Y_data[0 : n * feat_size])
                        T.block_attr({"sparse": True})
                        with T.init():
                            Y_data[vi * feat_size + vout] = T.float32(0)
                        Y_data[vi * feat_size + vout] = (
                            Y_data[vi * feat_size + vout]
                            + W_data[
                                (E_data[J_indptr[vi] + vj] * feat_size + vout) * feat_size + vin
                            ]
                            * X_data[J_indices[J_indptr[vi] + vj] * feat_size + vin]
                        )


@T.prim_func
def lowered_fused_reduction_4d_2d(
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
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    X_data = T.match_buffer(x, [nnz_l], dtype="float32")
    Y_data = T.match_buffer(y, [nnz_j], dtype="float32")
    J_indptr = T.match_buffer(indptr_j, [n + 1], dtype="int32")
    K_indptr = T.match_buffer(indptr_k, [nnz_j + 1], dtype="int32")
    L_indptr = T.match_buffer(indptr_l, [nnz_k + 1], dtype="int32")
    # body
    # with T.block("root")
    for v_vi, v_vj in T.grid(1, nnz_j):
        with T.block("reduction_4d_2d0"):
            vi, vj = T.axis.remap("SS", [v_vi, v_vj])
            T.reads(
                J_indptr[0 : n + 1],
                K_indptr[0 : nnz_j + 1],
                L_indptr[0 : nnz_k + 1],
                X_data[0:nnz_l],
                Y_data[0:nnz_j],
            )
            T.writes(Y_data[0:nnz_j])
            T.block_attr({"sparse": True})
            for v_vk in T.serial(K_indptr[vj + 1] - K_indptr[vj]):
                with T.block("reduction_4d_2d1"):
                    vk = T.axis.reduce(K_indptr[vj + 1] - K_indptr[vj], v_vk)
                    T.reads(
                        J_indptr[0 : n + 1],
                        K_indptr[0 : nnz_j + 1],
                        L_indptr[0 : nnz_k + 1],
                        X_data[0:nnz_l],
                        Y_data[0:nnz_j],
                    )
                    T.writes(Y_data[0:nnz_j])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Y_data[vj] = T.float32(0)
                    for v_vl in T.serial(
                        L_indptr[K_indptr[vj] + vk + 1] - L_indptr[K_indptr[vj] + vk]
                    ):
                        with T.block("reduction_4d_2d2"):
                            vl = T.axis.reduce(
                                L_indptr[K_indptr[vj] + vk + 1] - L_indptr[K_indptr[vj] + vk], v_vl
                            )
                            T.reads(
                                J_indptr[0 : n + 1],
                                K_indptr[0 : nnz_j + 1],
                                L_indptr[0 : nnz_k + 1],
                                X_data[0:nnz_l],
                                Y_data[0:nnz_j],
                            )
                            T.writes(Y_data[0:nnz_j])
                            T.block_attr({"sparse": True})
                            Y_data[vj] = Y_data[vj] + X_data[L_indptr[K_indptr[vj] + vk] + vl]


@T.prim_func
def lowered_fused_reduction_4d_3d(
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
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    X_data = T.match_buffer(x, [nnz_l], dtype="float32")
    Y_data = T.match_buffer(y, [nnz_k], dtype="float32")
    J_indptr = T.match_buffer(indptr_j, [n + 1], dtype="int32")
    K_indptr = T.match_buffer(indptr_k, [nnz_j + 1], dtype="int32")
    L_indptr = T.match_buffer(indptr_l, [nnz_k + 1], dtype="int32")
    # body
    # with T.block("root")
    for v_vi, v_vj, v_vk in T.grid(1, 1, nnz_k):
        with T.block("reduction_4d_3d0"):
            vi, vj, vk = T.axis.remap("SSS", [v_vi, v_vj, v_vk])
            T.reads(
                J_indptr[0 : n + 1],
                K_indptr[0 : nnz_j + 1],
                L_indptr[0 : nnz_k + 1],
                X_data[0:nnz_l],
                Y_data[0:nnz_k],
            )
            T.writes(Y_data[0:nnz_k])
            T.block_attr({"sparse": True})
            for v_vl in T.serial(L_indptr[vk + 1] - L_indptr[vk]):
                with T.block("reduction_4d_3d1"):
                    vl = T.axis.reduce(L_indptr[vk + 1] - L_indptr[vk], v_vl)
                    T.reads(
                        J_indptr[0 : n + 1],
                        K_indptr[0 : nnz_j + 1],
                        L_indptr[0 : nnz_k + 1],
                        X_data[0:nnz_l],
                        Y_data[0:nnz_k],
                    )
                    T.writes(Y_data[0:nnz_k])
                    T.block_attr({"sparse": True})
                    with T.init():
                        Y_data[vk] = T.float32(0)
                    Y_data[vk] = Y_data[vk] + X_data[L_indptr[vk] + vl]
