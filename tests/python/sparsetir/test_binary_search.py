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

from tvm.script import tir as T
from sparse_tir_lowered_iter_scripts import csrmm_dense_iter
import tvm

def test_binary_search():
    M, N, K, NNZ = csrmm_dense_iter.params[-4:]
    mod = tvm.IRModule.from_expr(csrmm_dense_iter.specialize({M: 128, N: 128, K: 128, NNZ: 1024}))
    mod = tvm.tir.transform.ExtractPreprocess()(mod)
    sch = tvm.tir.Schedule(mod)
    blk = sch.get_block("binary_search_block_0_0")
    i, j = sch.get_loops(blk) 
    sch.bind(i, "blockIdx.x") 
    sch.bind(j, "threadIdx.x")
    mod = tvm.tir.transform.LowerSparseBuffer()(sch.mod) 
    f = tvm.build(sch.mod["main"], target="cuda")
    print(f.imported_modules[0].get_source())

if __name__ == "__main__":
    test_binary_search()