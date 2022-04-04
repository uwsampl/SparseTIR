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
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
from tvm.script import tir as T
from lowered_tir import *


def test_csrmm():
    A = sp.random(512, 512, dtype="float32", density=0.0125, format="csr")
    x = np.random.rand(512, 128).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((512, 128)).astype("float32")

    n, m, k, nnz = lowered_csrmm.params[-4:]
    f = tvm.build(lowered_csrmm.specialize({n: 512, m: 512, k: 128, nnz: A.nnz}), target="llvm")

    ctx = tvm.cpu(0)
    A_indptr = tvm.nd.array(A.indptr.astype("int32"), device=ctx)
    A_indices = tvm.nd.array(A.indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(A.data.astype("float32"), device=ctx)
    X_nd = tvm.nd.array(x.reshape(-1), device=ctx)
    Y_nd = tvm.nd.array(y.reshape(-1), device=ctx)
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(y_ground_truth.reshape(-1), Y_nd.numpy(), rtol=1e-5, atol=1e-5)


def test_csr_reduce():
    A = sp.random(128, 128, dtype="float32", density=0.0125, format="csr")
    b_ground_truth = np.array(np.sum(A, axis=1))
    b = np.zeros((128,)).astype("float32")

    n, m, nnz = lowered_csr_reduce.params[-3:]
    f = tvm.build(lowered_csr_reduce.specialize({n: 128, m: 128, nnz: A.nnz}), target="llvm")

    ctx = tvm.cpu(0)
    A_indptr = tvm.nd.array(A.indptr.astype("int32"), device=ctx)
    A_indices = tvm.nd.array(A.indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(A.data.astype("float32"), device=ctx)
    B_nd = tvm.nd.array(b, device=ctx)
    f(A_data, B_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(b_ground_truth.reshape(-1), B_nd.numpy(), rtol=1e-5, atol=1e-5)


def test_csr_element_wise():
    A = sp.random(128, 128, dtype="float32", density=0.0125, format="csr")
    b_ground_truth = A * 2.5
    b = np.zeros((A.nnz,)).astype("float32")

    m, n, nnz = lowered_csr_element_wise.params[-3:]
    f = tvm.build(lowered_csr_element_wise.specialize({m: 128, n: 128, nnz: A.nnz}), target="llvm")

    ctx = tvm.cpu(0)
    A_indptr = tvm.nd.array(A.indptr.astype("int32"), device=ctx)
    A_indices = tvm.nd.array(A.indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(A.data.astype("float32"), device=ctx)
    B_nd = tvm.nd.array(b, device=ctx)
    f(A_data, B_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(b_ground_truth.data.reshape(-1), B_nd.numpy(), rtol=1e-5, atol=1e-5)


def test_bsrmm():
    block_size = 16
    nb = 32
    mb = 32
    feat_size = 256
    n = nb * block_size
    m = mb * block_size

    A_block = sp.random(mb, nb, dtype="float32", density=0.05, format="csr")
    indptr = A_block.indptr
    indices = A_block.indices
    nnzb = A_block.nnz
    data = np.random.rand(nnzb, block_size, block_size)
    A = sp.bsr_matrix((data, indices, indptr), shape=(n, m))
    x = np.random.rand(m, feat_size).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((n * feat_size,)).astype("float32")

    v_nb, v_mb, v_nnzb, v_blk, v_feat_size = lowered_bsrmm.params[-5:]
    f = tvm.build(
        lowered_bsrmm.specialize(
            {v_nb: nb, v_mb: mb, v_nnzb: nnzb, v_blk: block_size, v_feat_size: feat_size}
        ),
        target="llvm",
    )

    ctx = tvm.cpu(0)
    A_indptr = tvm.nd.array(indptr.astype("int32"), device=ctx)
    A_indices = tvm.nd.array(indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(data.reshape(-1).astype("float32"), device=ctx)
    X_nd = tvm.nd.array(x.reshape(-1), device=ctx)
    Y_nd = tvm.nd.array(y, device=ctx)
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(y_ground_truth.reshape(-1), Y_nd.numpy(), rtol=1e-5, atol=1e-5)


def test_ellmm():
    nnz_cols = 4
    nb = 64
    mb = 64
    feat_size = 1024
    nnz = nb * nnz_cols
    block_size = 16
    n = nb * block_size
    m = mb * block_size

    rng = np.random.default_rng()
    indptr = np.arange(0, (nb + 1) * nnz_cols, nnz_cols)
    indices = np.array([rng.choice(mb, size=nnz_cols, replace=False) for i in range(nb)])
    order = indices.argsort(axis=1)
    indices = np.array([indices[i, order[i]] for i in range(0, nb)]).reshape(-1)
    data = np.random.rand(nnz, block_size, block_size)
    A = sp.bsr_matrix((data, indices, indptr), shape=(n, m))
    x = np.random.rand(m, feat_size).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((n * feat_size,)).astype("float32")

    v_nb, v_mb, v_feat_size, v_col, v_blk = lowered_ellmm.params[-5:]
    f = tvm.build(
        lowered_ellmm.specialize(
            {
                v_nb: nb,
                v_mb: mb,
                v_feat_size: feat_size,
                v_col: nnz_cols,
                v_blk: block_size,
            }
        ),
        target="llvm",
    )

    ctx = tvm.cpu(0)
    A_indices = tvm.nd.array(indices.astype("int32"), device=ctx)
    A_data = tvm.nd.array(data.reshape(-1).astype("float32"), device=ctx)
    X_nd = tvm.nd.array(x.reshape(-1), device=ctx)
    Y_nd = tvm.nd.array(y, device=ctx)
    f(A_data, X_nd, Y_nd, A_indices)
    tvm.testing.assert_allclose(y_ground_truth.reshape(-1), Y_nd.numpy(), rtol=1e-5, atol=1e-5)


def test_sddmm():
    # generate random input
    m = 4096
    n = 4096
    k = 256
    C = sp.random(m, n, dtype="float32", density=0.0125, format="csr")
    indptr = C.indptr
    indices = C.indices
    C_coo = C.tocoo()
    nnz = C.nnz
    x = np.random.rand(m, k).astype("float32")
    y = np.random.rand(n, k).astype("float32")
    z_ground_truth = np.matmul(x, y.transpose())[C_coo.row, C_coo.col]
    z = np.zeros((nnz,)).astype("float32")

    # specialize function
    _, _, _, _, _, M, N, K, NNZ = lowered_sddmm.params
    sch = tir.Schedule(lowered_sddmm.specialize({M: m, N: n, K: k, NNZ: nnz}))
    blk_outer = sch.get_block("sddmm0")
    blk_inner = sch.get_block("sddmm1")
    (i,) = sch.get_loops(blk_outer)
    _, k = sch.get_loops(blk_inner)
    sch.bind(i, "blockIdx.x")
    sch.bind(k, "threadIdx.x")

    # convert numpy tensor to tvm ndarray
    C_indices = tvm.nd.array(indices.astype("int32"), device=tvm.cuda(0))
    C_indptr = tvm.nd.array(indptr.astype("int32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y.reshape(-1), device=tvm.cuda(0))
    C_data = tvm.nd.array(z, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod["main"], target="cuda")
    f(X_nd, Y_nd, C_data, C_indptr, C_indices)

    # assertion
    tvm.testing.assert_allclose(z_ground_truth, C_data.numpy(), rtol=1e-5)


def test_sddmm_fuse():
    # generate random input
    m = 4096
    n = 4096
    k = 256
    C = sp.random(m, n, dtype="float32", density=0.0125, format="csr")
    indptr = C.indptr
    indices = C.indices
    C_coo = C.tocoo()
    nnz = C.nnz
    x = np.random.rand(m, k).astype("float32")
    y = np.random.rand(n, k).astype("float32")
    z_ground_truth = np.matmul(x, y.transpose())[C_coo.row, C_coo.col]
    z = np.zeros((nnz,)).astype("float32")

    # specialize function
    _, _, _, _, _, M, N, K, NNZ = lowered_sddmm_fuse.params
    sch = tir.Schedule(lowered_sddmm_fuse.specialize({M: m, N: n, K: k, NNZ: nnz}))
    blk = sch.get_block("sddmm0")
    i, j, k = sch.get_loops(blk)
    sch.unroll(i)
    sch.bind(j, "blockIdx.x")
    sch.bind(k, "threadIdx.x")

    # convert numpy tensor to tvm ndarray
    C_indices = tvm.nd.array(indices.astype("int32"), device=tvm.cuda(0))
    C_indptr = tvm.nd.array(indptr.astype("int32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y.reshape(-1), device=tvm.cuda(0))
    C_data = tvm.nd.array(z, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod["main"], target="cuda")
    f(X_nd, Y_nd, C_data, C_indptr, C_indices)

    # assertion
    tvm.testing.assert_allclose(z_ground_truth, C_data.numpy(), rtol=1e-5)


def test_bmm():
    # generate random input
    batch_size = 32
    n_arr = np.random.randint(128, 1024, size=(batch_size,)).astype("int32")
    m_arr = np.random.randint(128, 1024, size=(batch_size,)).astype("int32")
    k_arr = np.random.randint(128, 1024, size=(batch_size,)).astype("int32")
    nm_arr = n_arr * m_arr
    mk_arr = m_arr * k_arr
    nk_arr = n_arr * k_arr
    indptr_n = np.concatenate(([0], n_arr)).cumsum()
    indptr_m = np.concatenate(([0], m_arr)).cumsum()
    indptr_k = np.concatenate(([0], k_arr)).cumsum()
    indptr_nm = np.concatenate(([0], nm_arr)).cumsum()
    indptr_mk = np.concatenate(([0], mk_arr)).cumsum()
    indptr_nk = np.concatenate(([0], nk_arr)).cumsum()
    nnz_i = indptr_n[-1]
    nnz_j = indptr_m[-1]
    nnz_k = indptr_k[-1]
    nnz_ij = indptr_nm[-1]
    nnz_jk = indptr_mk[-1]
    nnz_ik = indptr_nk[-1]
    As = [np.random.rand(n, m).astype("float32") for n, m in zip(n_arr, m_arr)]
    Bs = [np.random.rand(m, k).astype("float32") for m, k in zip(m_arr, k_arr)]
    Cs = [np.matmul(A, B) for A, B in zip(As, Bs)]
    A_flatten = np.concatenate([A.flatten() for A in As], 0)
    B_flatten = np.concatenate([B.flatten() for B in Bs], 0)
    c_flatten = np.concatenate([C.flatten() for C in Cs], 0)

    # specialize function
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        BATCH,
        NNZ_I,
        NNZ_J,
        NNZ_K,
        NNZ_IJ,
        NNZ_JK,
        NNZ_IK,
    ) = lowered_bmm.params
    sch = tir.Schedule(
        lowered_bmm.specialize(
            {
                BATCH: batch_size,
                NNZ_I: nnz_i,
                NNZ_J: nnz_j,
                NNZ_K: nnz_k,
                NNZ_IJ: nnz_ij,
                NNZ_JK: nnz_jk,
                NNZ_IK: nnz_ik,
            }
        )
    )
    bmm_outer = sch.get_block("bmm0")
    (b,) = sch.get_loops(bmm_outer)
    bmm_inner = sch.get_block("bmm1")
    i, j, k = sch.get_loops(bmm_inner)
    sch.reorder(i, k, j)
    io, ii = sch.split(i, [None, 32])
    ko, ki = sch.split(k, [None, 32])
    sch.bind(b, "blockIdx.x")
    sch.bind(ki, "threadIdx.x")
    sch.bind(ii, "threadIdx.y")
    sch.decompose_reduction(bmm_inner, j)

    # convert numpy tensor to tvm ndarray
    dev = tvm.cuda(0)
    A_nd = tvm.nd.array(A_flatten, device=dev)
    B_nd = tvm.nd.array(B_flatten, device=dev)
    C_nd = tvm.nd.array(np.zeros_like(c_flatten), device=dev)
    indptr_n_nd = tvm.nd.array(indptr_n.astype("int32"), device=dev)
    indptr_m_nd = tvm.nd.array(indptr_m.astype("int32"), device=dev)
    indptr_k_nd = tvm.nd.array(indptr_k.astype("int32"), device=dev)
    indptr_nm_nd = tvm.nd.array(indptr_nm.astype("int32"), device=dev)
    indptr_mk_nd = tvm.nd.array(indptr_mk.astype("int32"), device=dev)
    indptr_nk_nd = tvm.nd.array(indptr_nk.astype("int32"), device=dev)

    # build function
    f = tvm.build(sch.mod["main"], target="cuda")
    f(
        A_nd,
        B_nd,
        C_nd,
        indptr_n_nd,
        indptr_m_nd,
        indptr_k_nd,
        indptr_nm_nd,
        indptr_mk_nd,
        indptr_nk_nd,
    )

    # assertion
    tvm.testing.assert_allclose(C_nd.numpy(), c_flatten, rtol=1e-5)


def test_square_sum():
    density = 0.0125
    M = N1 = N2 = 128
    A_J = sp.random(M, N1, dtype="float32", density=1 - (1 - density) ** N2, format="csr")
    indptr_j = A_J.indptr
    indices_j = A_J.indices
    nnz_j = A_J.nnz
    A_K = sp.random(nnz_j, N2, dtype="float32", density=density, format="csr")
    indptr_k = A_K.indptr
    indices_k = A_K.indices
    nnz_k = A_K.nnz
    data = A_K.data

    b_ij = np.asarray(A_K.sum(axis=1)).squeeze()
    A_J = sp.csr_matrix((b_ij, indices_j, indptr_j), shape=(M, N1))
    b_ground_truth = np.asarray(A_J.sum(axis=1)).squeeze()
    b = np.zeros((M,)).astype("float32")

    v_nnz_j, v_nnz_k, v_M, v_N1, v_N2 = lowered_square_sum.params[-5:]
    f = tvm.build(
        lowered_square_sum.specialize({v_nnz_j: nnz_j, v_nnz_k: nnz_k, v_M: M, v_N1: N1, v_N2: N2}),
        target="llvm",
    )

    ctx = tvm.cpu(0)
    A_data = tvm.nd.array(data.astype("float32"), device=ctx)
    A_indptr_j = tvm.nd.array(indptr_j.astype("int32"), device=ctx)
    A_indices_j = tvm.nd.array(indices_j.astype("int32"), device=ctx)
    A_indptr_k = tvm.nd.array(indptr_k.astype("int32"), device=ctx)
    A_indices_k = tvm.nd.array(indices_k.astype("int32"), device=ctx)
    B_data = tvm.nd.array(b.astype("float32"), device=ctx)
    f(A_data, B_data, A_indptr_j, A_indices_j, A_indptr_k, A_indices_k)

    tvm.testing.assert_allclose(b_ground_truth, B_data.numpy(), rtol=1e-5, atol=1e-5)


def test_square_sum_two_K():
    sch = tir.Schedule(lowered_square_sum_two_K, debug_mask="all")
    (i,) = sch.get_loops(sch.get_block("square_sum_2"))
    sch.bind(i, "threadIdx.x")

    density = 0.0125
    M = N1 = N2 = 128
    A_J = sp.random(M, N1, dtype="float32", density=1 - (1 - density) ** N2, format="csr")
    indptr_j = A_J.indptr
    indices_j = A_J.indices
    nnz_j = A_J.nnz
    A_K = sp.random(nnz_j, N2, dtype="float32", density=density, format="csr")
    indptr_k = A_K.indptr
    indices_k = A_K.indices
    nnz_k = A_K.nnz
    data = A_K.data

    b_ij = np.asarray(A_K.sum(axis=1)).squeeze()
    A_J = sp.csr_matrix((b_ij, indices_j, indptr_j), shape=(M, N1))
    b_ground_truth = np.asarray(A_J.sum(axis=1)).squeeze()
    b = np.zeros((M,)).astype("float32")

    v_nnz_j, v_nnz_k, v_M, v_N1, v_N2 = sch.mod["main"].params[-5:]
    f = tvm.build(
        sch.mod["main"].specialize({v_nnz_j: nnz_j, v_nnz_k: nnz_k, v_M: M, v_N1: N1, v_N2: N2}),
        target="cuda",
    )

    ctx = tvm.device("cuda")
    A_data = tvm.nd.array(data.astype("float32"), device=ctx)
    A_indptr_j = tvm.nd.array(indptr_j.astype("int32"), device=ctx)
    A_indices_j = tvm.nd.array(indices_j.astype("int32"), device=ctx)
    A_indptr_k0 = tvm.nd.array(indptr_k.astype("int32"), device=ctx)
    A_indices_k0 = tvm.nd.array(indices_k.astype("int32"), device=ctx)
    A_indptr_k1 = tvm.nd.array(indptr_k.astype("int32"), device=ctx)
    A_indices_k1 = tvm.nd.array(indices_k.astype("int32"), device=ctx)
    B_data = tvm.nd.array(b.astype("float32"), device=ctx)
    f(A_data, B_data, A_indptr_j, A_indices_j, A_indptr_k0, A_indices_k0, A_indptr_k1, A_indices_k1)

    tvm.testing.assert_allclose(b_ground_truth, B_data.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_csrmm()
    test_csr_reduce()
    test_csr_element_wise()
    test_bsrmm()
    test_ellmm()
    test_sddmm()
    test_sddmm_fuse()
    test_bmm()
    test_square_sum()
    test_square_sum_two_K()
