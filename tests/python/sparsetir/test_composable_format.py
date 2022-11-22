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
import numpy as np
from tvm.sparse import FormatRewriteRule, format_decompose
from sparse_tir_scripts import csrmm
from sparse_tir_composable_format_scripts import (
    bsr,
    bsr_rewrite_with_preprocess,
    ell,
    ell_rewrite_with_preprocess,
    padding,
    padding_rewrite_with_preprocess,
)


def csr2bsr_inv_index_map(block_size):
    def func(io, jo, ii, ji):
        return io * block_size + ii, jo * block_size + ji

    return func


def csr2bsr_index_map(block_size):
    def func(i, j):
        return i // block_size, j // block_size, i % block_size, j % block_size

    return func


def csr2ell_inv_index_map(o, i, j):
    return i, j


def csr2ell_index_map(i, j):
    return 0, i, j


def test_csrmm_bsr_rewrite():
    block_size_symbol = bsr.params[-1]
    rewrites = []
    for block_size in [4, 16, 32]:
        rewrites.append(
            FormatRewriteRule(
                str(block_size),
                bsr.specialize({block_size_symbol: block_size}),
                ["A"],
                ["I", "J"],
                ["IO", "JO", "II", "JI"],
                {"I": ["IO", "II"], "J": ["JO", "JI"]},
                csr2bsr_index_map(block_size),
                csr2bsr_inv_index_map(block_size),
            )
        )
    mod = tvm.IRModule.from_expr(csrmm)
    mod = format_decompose(mod, rewrites)
    print(mod["main"].script())
    tvm.ir.assert_structural_equal(mod["main"], bsr_rewrite_with_preprocess, True)


def test_csrmm_ell_rewrite():
    nnz_cols_symbol = ell.params[-1]
    rewrites = []
    for nnz_cols in [4, 8, 16, 32, 64, 128, 512]:
        rewrites.append(
            FormatRewriteRule(
                str(nnz_cols),
                ell.specialize({nnz_cols_symbol: nnz_cols}),
                ["A"],
                ["I", "J"],
                ["O", "I", "J"],
                {"I": ["O", "I"], "J": ["J"]},
                csr2ell_index_map,
                csr2ell_inv_index_map,
            )
        )
    mod = tvm.IRModule.from_expr(csrmm)
    mod = format_decompose(mod, rewrites)
    tvm.ir.assert_structural_equal(mod["main"], ell_rewrite_with_preprocess, True)


def csrpadding_inv_index_map(i, jo, ji):
    return i, ji


def csrpadding_index_map(i, j):
    return i, 0, j


def test_csrmm_padding_rewrite():
    pad_size_symbol = padding.params[-1]
    pad_size = 32
    rewrites = [
        FormatRewriteRule(
            str(pad_size),
            padding.specialize({pad_size_symbol: pad_size}),
            ["A"],
            ["I", "J"],
            ["I", "JO", "JI"],
            {"I": ["I"], "J": ["JO", "JI"]},
            csrpadding_index_map,
            csrpadding_inv_index_map,
        )
    ]
    mod = tvm.IRModule.from_expr(csrmm)
    mod = format_decompose(mod, rewrites)
    tvm.ir.assert_structural_equal(mod["main"], padding_rewrite_with_preprocess, True)


if __name__ == "__main__":
    test_csrmm_bsr_rewrite()
    test_csrmm_ell_rewrite()
    test_csrmm_padding_rewrite()
