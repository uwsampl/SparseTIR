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
from tvm.sparse import FormatRewriteRule
from sparse_tir_scripts import csrmm
from sparse_tir_format_rewrite_scripts import bsr


def csr2bsr_index_map(block_size):
    def func(i, j):
        return i // block_size, j // block_size, i % block_size, j % block_size

    return func


def csr2bsr_inv_index_map(block_size):
    def func(io, jo, ii, ji):
        return io * block_size + ii, jo * block_size + ji

    return func


def test_declare_format_rewrite_rule():
    csr2bsr_32 = FormatRewriteRule(
        "csr2bsr",
        bsr,
        ["A"],
        {"I": ["IO", "II"], "J": ["JO", "JI"]},
        csr2bsr_index_map(32),
        csr2bsr_inv_index_map(32),
    )
    print(csr2bsr_32)
    print(csr2bsr_32.name)
    print(csr2bsr_32.new_format_desc)
    print(csr2bsr_32.buffers_to_rewrite)
    print(csr2bsr_32.axis_map)
    print(csr2bsr_32.idx_map)
    print(csr2bsr_32.inv_idx_map)


def test_csrmm_bsr_rewrite():
    block_size_symbol = bsr.params[-1]
    rewrites = []
    for block_size in [4, 16, 32]:
        rewrites.append(
            FormatRewriteRule(
                "csr2bsr_{}".format(block_size),
                bsr.specialize({block_size_symbol: block_size}),
                ["A"],
                {"I": ["IO", "II"], "J": ["JO", "JI"]},
                csr2bsr_index_map(block_size),
                csr2bsr_inv_index_map(block_size),
            )
        )
    mod = tvm.IRModule.from_expr(csrmm)
    mod = tvm.tir.transform.SparseFormatRewrite(rewrites)(mod)
    print(mod["main"].script())


if __name__ == "__main__":
    test_declare_format_rewrite_rule()
    test_csrmm_bsr_rewrite()
