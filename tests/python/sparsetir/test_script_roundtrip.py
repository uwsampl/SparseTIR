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
import sparse_tir_lowered_iter_scripts
import sparse_tir_lowered_buffer_scripts
import sparse_tir_scripts
import sparse_tir_composable_format_scripts


func_name_list = [
    "csrmm",
    "csrmm_dense_iter",
    "segment_reduce",
    "csr_reduce",
    "bsrmm",
    "ellmm",
    "csr_element_wise",
    "bmm",
    "sddmm",
    "fused_sddmm",
    "square_sum",
    "square_sum_two_K",
    "fused_reduction_4d_2d",
    "fused_reduction_4d_3d",
    "rgcn_homo_forward",
    "rgcn_hetero_forward",
    "sparse_softmax",
    "csr2bsr",
]


def specialize_csrmm(f):
    M, N, K, NNZ = f.params[-4:]
    return f.specialize({M: 128, N: 128, K: 128, NNZ: 1024})


def specialize_csrmm_dense_iter(f):
    M, N, K, NNZ = f.params[-4:]
    return f.specialize({M: 128, N: 128, K: 128, NNZ: 1024})


def specialize_segment_reduce(f):
    N, NNZ = f.params[-2:]
    return f.specialize({N: 128, NNZ: 1024})


def specialize_csr_reduce(f):
    N, M, NNZ = f.params[-3:]
    return f.specialize({N: 128, M: 128, NNZ: 1024})


def specialize_bsrmm(f):
    NB, MB, NNZB, BLK, FEAT_SIZE = f.params[-5:]
    return f.specialize({NB: 16, MB: 16, NNZB: 128, BLK: 32, FEAT_SIZE: 256})


def specialize_ellmm(f):
    NB, MB, FEAT_SIZE, COL, BLK = f.params[-5:]
    return f.specialize({NB: 16, MB: 16, FEAT_SIZE: 256, COL: 4, BLK: 32})


def specialize_csr_element_wise(f):
    M, N, NNZ = f.params[-3:]
    return f.specialize({M: 128, N: 128, NNZ: 1024})


def specialize_bmm(f):
    BATCH_SIZE, NNZ_I, NNZ_J, NNZ_K, NNZ_IJ, NNZ_JK, NNZ_IK = f.params[-7:]
    return f.specialize(
        {
            BATCH_SIZE: 32,
            NNZ_I: 128,
            NNZ_J: 128,
            NNZ_K: 128,
            NNZ_IJ: 1024,
            NNZ_JK: 1024,
            NNZ_IK: 1024,
        }
    )


def specialize_sddmm(f):
    M, N, K, NNZ = f.params[-4:]
    return f.specialize({M: 128, N: 128, K: 128, NNZ: 1024})


def specialize_fused_sddmm(f):
    M, N, K, NNZ = f.params[-4:]
    return f.specialize({M: 128, N: 128, K: 128, NNZ: 1024})


def specialize_square_sum(f):
    NNZ_J, NNZ_K, M, N1, N2 = f.params[-5:]
    return f.specialize({NNZ_J: 128, NNZ_K: 1024, M: 16, N1: 16, N2: 16})


def specialize_square_sum_two_K(f):
    NNZ_J, NNZ_K, M, N1, N2 = f.params[-5:]
    return f.specialize({NNZ_J: 128, NNZ_K: 1024, M: 16, N1: 16, N2: 16})


def specialize_fused_reduction_4d_2d(f):
    N, NNZ_J, NNZ_K, NNZ_L = f.params[-4:]
    return f.specialize({N: 16, NNZ_J: 128, NNZ_K: 256, NNZ_L: 1024})


def specialize_fused_reduction_4d_3d(f):
    N, NNZ_J, NNZ_K, NNZ_L = f.params[-4:]
    return f.specialize({N: 16, NNZ_J: 128, NNZ_K: 256, NNZ_L: 1024})


def specialize_rgcn_homo_forward(f):
    M, N, R, FEAT_SIZE, NNZ = f.params[-5:]
    return f.specialize({M: 128, N: 128, R: 16, FEAT_SIZE: 128, NNZ: 1024})


def specialize_rgcn_hetero_forward(f):
    M, N, R, FEAT_SIZE, NNZ_I, NNZ_J = f.params[-6:]
    return f.specialize({M: 128, N: 128, R: 16, FEAT_SIZE: 128, NNZ_I: 32, NNZ_J: 1024})


def specialize_sparse_softmax(f):
    N, NNZ = f.params[-2:]
    return f.specialize({N: 128, NNZ: 1024})


def specialize_csr2bsr(f):
    M_in, N_in, M_out, N_out, NNZ_in, NNZ_out, BLK_SIZE = f.params[-7:]
    return f.specialize(
        {M_in: 1024, N_in: 1024, M_out: 32, N_out: 32, NNZ_in: 16384, NNZ_out: 128, BLK_SIZE: 32}
    )


def specialize(funcname, f):
    return globals()["specialize_" + funcname](f)


def test_sparse_tir_scripts():
    for func_name in func_name_list:
        func = getattr(sparse_tir_scripts, func_name)
        rt_func = tvm.script.from_source(func.script(show_meta=True))
        tvm.ir.assert_structural_equal(func, rt_func, True)


def test_sparse_tir_scripts_specialize():
    for func_name in func_name_list:
        func = getattr(sparse_tir_scripts, func_name)
        specialized_func = specialize(func_name, func)
        rt_func = tvm.script.from_source(specialized_func.script(show_meta=True))
        tvm.ir.assert_structural_equal(specialized_func, rt_func, True)


def test_sparse_tir_lowered_iter_scripts():
    for func_name in func_name_list:
        func = getattr(sparse_tir_lowered_iter_scripts, func_name)
        rt_func = tvm.script.from_source(func.script(show_meta=True))
        tvm.ir.assert_structural_equal(func, rt_func, True)


def test_sparse_tir_lowered_buffer_scripts():
    for func_name in func_name_list:
        func = getattr(sparse_tir_lowered_buffer_scripts, func_name)
        rt_func = tvm.script.from_source(func.script(show_meta=True))
        tvm.ir.assert_structural_equal(func, rt_func, True)


def test_sparse_tir_composable_format_scripts():
    for func_name in ["bsr_rewrite_with_preprocess"]:
        func = getattr(sparse_tir_composable_format_scripts, func_name)
        rt_func = tvm.script.from_source(func.script(show_meta=True))
        tvm.ir.assert_structural_equal(func, rt_func, True)


if __name__ == "__main__":
    test_sparse_tir_scripts()
    test_sparse_tir_scripts_specialize()
    test_sparse_tir_lowered_iter_scripts()
    test_sparse_tir_composable_format_scripts()
    test_sparse_tir_lowered_buffer_scripts()
