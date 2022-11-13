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
import sparse_tir_scripts
import sparse_tir_lowered_iter_scripts
from tvm.testing.utils import exclude_targets
from tvm.sparse import lower_sparse_iter


func_name_list = [
    "csrmm",
    "csrmm_dense_iter",
    "segment_reduce",
    "csr_reduce",
    "bsrmm",
    "ellmm",
    "csr_element_wise",
    "hyper_gnn",
    # "bmm",
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


def test_sparse_tir_lower_iter():
    for func_name in func_name_list:
        print(func_name)
        mod = tvm.IRModule.from_expr(getattr(sparse_tir_scripts, func_name))
        mod = lower_sparse_iter(mod)
        print(mod["main"].script())
        lowered_func = getattr(sparse_tir_lowered_iter_scripts, func_name)
        tvm.ir.assert_structural_equal(mod["main"], lowered_func, True)


if __name__ == "__main__":
    test_sparse_tir_lower_iter()
