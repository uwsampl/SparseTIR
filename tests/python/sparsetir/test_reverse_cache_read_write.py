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
import argparse
from tvm.script import tir as T
from tvm.sparse import lower_sparse_buffer, lower_sparse_iter


@T.prim_func
def tcspmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    mb: T.int32,
    nb: T.int32,
    nnzb: T.int32,
    feat_size: T.int32,
    block_size: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    IO = T.dense_fixed(mb)
    JO = T.dense_variable(IO, (nb, nnzb), indptr, "int32")
    II = T.dense_fixed(block_size)
    JI = T.sparse_fixed(JO, (nb * block_size, block_size), indices, "int32")
    J = T.dense_fixed(nb * block_size)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, [IO, JO, II, JI], "float16")
    B = T.match_sparse_buffer(b, [J, F], "float16")
    C = T.match_sparse_buffer(c, [IO, II, F], "float16")
    with T.iter([IO, JO, II, JI, F], "SRSRS", "tcspmm") as [io, jo, ii, ji, f]:
        with T.init():
            C[io, ii, f] = T.float16(0)
        C[io, ii, f] = (
            C[io, ii, f] + A[io, jo, ii, ji] * B[ji, f]
        )


def test_tc_spmm_cache_read():
    MB, NB, NNZB, F, B = tcspmm.params[-5:]
    mod = tvm.IRModule.from_expr(tcspmm.specialize({
        MB: 128, NB: 128, NNZB: 1024, F: 64, B: 16
    }))
    mod = lower_sparse_iter(mod)
    sch = tir.Schedule(mod)
    blk_outer = sch.get_block("tcspmm0")
    blk_inner = sch.get_block("tcspmm1")
    i, = sch.get_loops(blk_outer)
    jo, ii, ji, f = sch.get_loops(blk_inner)
    fo, fi = sch.split(f, [None, 16])
    sch.reorder(fo, ii, ji, fi)
    new_blk = sch.blockize(ii)
    print(sch.mod["main"].script())
    B_local = sch.reverse_cache_read(blk_inner, 2, "shared")
    print(sch.mod["main"].script())


if __name__ == "__main__":
    test_tc_spmm_cache_read()