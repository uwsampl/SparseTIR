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
import pytest
from lowered_tir import *
from sparse_tir_scripts import *


def test_csrmm():
    mod = tvm.IRModule.from_expr(csrmm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_csrmm, True)


def test_csrmm_dense_iter():
    mod = tvm.IRModule.from_expr(csrmm_dense_iter)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_csrmm_dense_iter, True)


def test_segment_reduce():
    mod = tvm.IRModule.from_expr(segment_reduce)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_segment_reduce, True)


def test_csr_reduce():
    mod = tvm.IRModule.from_expr(csr_reduce)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_csr_reduce, True)


def test_bsrmm():
    mod = tvm.IRModule.from_expr(bsrmm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_bsrmm, True)


def test_ellpack_mm():
    mod = tvm.IRModule.from_expr(ellmm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_ellmm, True)


def test_csr_element_wise():
    mod = tvm.IRModule.from_expr(csr_element_wise)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_csr_element_wise, True)


def test_bmm():
    mod = tvm.IRModule.from_expr(bmm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_bmm)


def test_sddmm():
    mod = tvm.IRModule.from_expr(sddmm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_sddmm)


def test_fused_sddmm():
    mod = tvm.IRModule.from_expr(fused_sddmm)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_sddmm_fuse)


def test_square_sum():
    mod = tvm.IRModule.from_expr(square_sum)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_square_sum, True)


def test_square_sum_two_K():
    mod = tvm.IRModule.from_expr(square_sum_two_K)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_square_sum_two_K, True)


def test_fused_reduction():
    mod = tvm.IRModule.from_expr(fused_reduction_4d_2d)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_fused_reduction_4d_2d, True)

    mod = tvm.IRModule.from_expr(fused_reduction_4d_3d)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    tvm.ir.assert_structural_equal(mod["main"], lowered_fused_reduction_4d_3d, True)


if __name__ == "__main__":
    test_csrmm()
    test_csrmm_dense_iter()
    test_segment_reduce()
    test_csr_reduce()
    test_bsrmm()
    test_ellpack_mm()
    test_csr_element_wise()
    test_sddmm()
    test_fused_sddmm()
    test_bmm()
    test_square_sum()
    test_square_sum_two_K()
    test_fused_reduction()
