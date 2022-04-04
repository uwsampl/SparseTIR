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
from scipy.sparse import bsr
import pytest


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), K), "float32")
    C = T.match_sparse_buffer(c, (I, K), "float32")
    with T.iter([I, K, J], "SSR", "csrmm") as [vi, vk, vj]:
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


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
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I,), "float32")
    with T.iter([I, J], "SR", "csr_reduce") as [vi, vj]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj]


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
    I = T.dense_fixed(nb)
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), "int32")
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), BJ, F), "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([I, J, BI, BJ, F], "SRSRS", "bsrmm") as [
        vi,
        vj,
        vbi,
        vbj,
        vf,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


@T.prim_func
def ellpack_mm(
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
    I = T.dense_fixed(nb)
    J = T.sparse_fixed(I, (mb, col), indices, "int32")
    F = T.dense_fixed(feat_size)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), BJ, F), "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([I, J, BI, BJ, F], "SRSRS", "ellmm") as [
        vi,
        vj,
        vbi,
        vbj,
        vf,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


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
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (I, J), "float32")

    with T.iter([I, J], "SS", "csr_element_wise") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.5


@T.prim_func
def reordered_bsrmm(
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
    I = T.dense_fixed(nb)
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), "int32")
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float32")
    B = T.match_sparse_buffer(b, (T.dense(J), BJ, F), "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([BI, BJ, I, J, F], "SRSRS", "bsrmm") as [
        vbi,
        vbj,
        vi,
        vj,
        vf,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


def test_get_sparse_iteration():
    sch = tir.Schedule(csrmm, debug_mask="all")
    block_rv = sch.get_sparse_iteration("csrmm")
    block = sch.get(block_rv)
    assert block.name == "csrmm"
    assert block.same_as(csrmm.body)


def test_get_sp_iters():
    sch = tir.Schedule(csrmm, debug_mask="all")
    block = sch.get_sparse_iteration("csrmm")
    vi, vj, vk = sch.get_sp_iters(block)
    assert vi.same_as(csrmm.body.sp_iter_vars[0])
    assert vj.same_as(csrmm.body.sp_iter_vars[1])
    assert vk.same_as(csrmm.body.sp_iter_vars[2])


def test_reorder():
    sch = tir.Schedule(bsrmm, debug_mask="all")
    block = sch.get_sparse_iteration("bsrmm")
    i, j, bi, bj, f = sch.get_sp_iters(block)
    sch.sparse_reorder(block, [bi, bj, i, j, f])
    tvm.ir.assert_structural_equal(sch.mod["main"], reordered_bsrmm, True)
    assert sch.get(block).name == "bsrmm"


def test_reorder_fail_on_dependency():
    sch = tir.Schedule(bsrmm, debug_mask="all")
    block = sch.get_sparse_iteration("bsrmm")
    i, j, bi, bj, f = sch.get_sp_iters(block)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.sparse_reorder(block, [bi, bj, j, i, f])


def test_reorder_fail_on_new_order_length():
    sch = tir.Schedule(bsrmm, debug_mask="all")
    block = sch.get_sparse_iteration("bsrmm")
    i, j, bi, bj, f = sch.get_sp_iters(block)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.sparse_reorder(block, [bi, bj, i, j])


if __name__ == "__main__":
    test_get_sparse_iteration()
    test_get_sp_iters()
    test_reorder()
    test_reorder_fail_on_dependency()
    test_reorder_fail_on_new_order_length()
