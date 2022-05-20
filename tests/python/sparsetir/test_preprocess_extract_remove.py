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

from sparse_tir_format_rewrite_scripts import bsr_rewrite_with_preprocess
from sparse_tir_lowered_iter_scripts import fused_sddmm
import tvm


def test_extract_preprocess_bsr_rewrite():
    mod = tvm.IRModule.from_expr(bsr_rewrite_with_preprocess)
    mod_preprocess = tvm.tir.transform.ExtractPreprocess()(mod)
    print(mod_preprocess["main"].script())


def test_extract_preprocess_fused_sddmm():
    mod = tvm.IRModule.from_expr(fused_sddmm)
    mod_preprocess = tvm.tir.transform.ExtractPreprocess()(mod)
    print(mod_preprocess["main"].script())


def test_remove_preprocess_bsr_rewrite():
    mod = tvm.IRModule.from_expr(bsr_rewrite_with_preprocess)
    mod_preprocess = tvm.tir.transform.RemovePreprocess()(mod)
    print(mod_preprocess["main"].script())


def test_remove_preprocess_fused_sddmm():
    mod = tvm.IRModule.from_expr(fused_sddmm)
    mod_preprocess = tvm.tir.transform.RemovePreprocess()(mod)
    print(mod_preprocess["main"].script())


if __name__ == "__main__":
    test_extract_preprocess_bsr_rewrite()
    test_extract_preprocess_fused_sddmm()
    test_remove_preprocess_bsr_rewrite()
    test_remove_preprocess_fused_sddmm()
