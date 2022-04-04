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
from sparse_tir_scripts import fused_sddmm
import sparse_tir_lowered_iter_scripts
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer

def test_fused_sddmm():
    mod = tvm.IRModule.from_expr(fused_sddmm)
    mod = lower_sparse_iter(mod)
    print(mod["main"].script())
    mod = lower_sparse_buffer(mod)
    print(mod["main"].script())

if __name__ == "__main__":
    test_fused_sddmm()

