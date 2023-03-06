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
from sparse_tir_scripts import csrmm, bsrmm, sddmm, fused_sddmm


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
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(nb)
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), "int32")
    J_detach = T.dense_fixed(mb)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float32")
    B = T.match_sparse_buffer(b, (J_detach, BJ, F), "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.sp_iter([BI, BJ, I, J, F], "SRSRS", "bsrmm") as [
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
    sp_iteration_rv = sch.get_sparse_iteration("csrmm")
    sp_iteration = sch.get(sp_iteration_rv)
    assert sp_iteration.name == "csrmm"
    assert sp_iteration.same_as(csrmm.body.block.body)
    sch.annotate(sp_iteration_rv, "guard", True)
    new_sp_iteration = sch.get(sp_iteration_rv)
    assert new_sp_iteration.annotations.get("guard") == True


def test_get_sp_iters():
    sch = tir.Schedule(csrmm, debug_mask="all")
    block = sch.get_sparse_iteration("csrmm")
    vi, vj, vk = sch.get_sp_iters(block)
    assert vi.same_as(csrmm.body.block.body.sp_iter_vars[0])
    assert vj.same_as(csrmm.body.block.body.sp_iter_vars[1])
    assert vk.same_as(csrmm.body.block.body.sp_iter_vars[2])


def test_reorder():
    sch = tir.Schedule(bsrmm, debug_mask="all")
    block = sch.get_sparse_iteration("bsrmm")
    i, bi, bj, f, j = sch.get_sp_iters(block)
    sch.sparse_reorder(block, [bi, bj, i, j, f])
    tvm.ir.assert_structural_equal(sch.mod["main"], reordered_bsrmm, True)
    assert sch.get(block).name == "bsrmm"


def test_fuse():
    sch = tir.Schedule(sddmm, debug_mask="all")
    block = sch.get_sparse_iteration("sddmm")
    i, j, k = sch.get_sp_iters(block)
    sch.sparse_fuse(block, [i, j])
    tvm.ir.assert_structural_equal(sch.mod["main"], fused_sddmm)


def test_reorder_fail_on_dependency():
    sch = tir.Schedule(bsrmm, debug_mask="all")
    block = sch.get_sparse_iteration("bsrmm")
    i, bi, bj, f, j = sch.get_sp_iters(block)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.sparse_reorder(block, [bi, bj, j, i, f])


def test_reorder_fail_on_new_order_length():
    sch = tir.Schedule(bsrmm, debug_mask="all")
    block = sch.get_sparse_iteration("bsrmm")
    i, bi, bj, f, j = sch.get_sp_iters(block)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.sparse_reorder(block, [bi, bj, i, j])


if __name__ == "__main__":
    test_get_sparse_iteration()
    test_get_sp_iters()
    test_reorder()
    test_fuse()
    test_reorder_fail_on_dependency()
    test_reorder_fail_on_new_order_length()
