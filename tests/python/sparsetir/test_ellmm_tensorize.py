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
import tvm.tir as tir
from tvm.script import tir as T
from sparse_tir_scripts import ellmm
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer


@T.prim_func
def wmma_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1, scope="local")
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1, scope="local")
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1, scope="local")

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, k, j in T.grid(16, 16, 16):
            with T.block("update"):
                vi, vk, vj = T.axis.remap("SRS", [i, k, j])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def wmma_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), align=128, offset_factor=1, scope="local")
    B = T.match_buffer(b, (16, 16), align=128, offset_factor=1, scope="local")
    C = T.match_buffer(c, (16, 16), align=128, offset_factor=1, scope="local")

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(
            T.tvm_mma_sync(
                C.data,
                C.elem_offset // 256,
                A.data,
                A.elem_offset // 256,
                B.data,
                B.elem_offset // 256,
                C.data,
                C.elem_offset // 256,
                dtype="handle",
            )
        )

tir.TensorIntrin.register("wmma_intrin", wmma_desc, wmma_intrin)


def test_blocked_ellmm_tensorize():
    NB, MB, FEAT_SIZE, COL, BLK = ellmm.params[-5:]
    mod = tvm.IRModule.from_expr(
        ellmm.specialize({NB: 32, MB: 32, FEAT_SIZE: 128, COL: 2, BLK: 16})
    )
    mod = lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    blk = sch.get_block("ellmm0")
    i, j, bi, bj, f = sch.get_loops(blk)
    fo, fi = sch.split(f, [None, 16])
    sch.reorder(i, j, fo, bi, bj, fi)
    blk_inner = sch.blockize(bi)
    blk, blk_inner = blk_inner, blk
    A_local = sch.cache_read(blk_inner, 1, "local")
    B_local = sch.cache_read(blk_inner, 2, "local")
    C_local = sch.cache_write(blk_inner, 0, "local")
    sch.hide_buffer_access(blk_inner, "read", [3])
    sch.tensorize(bi, "wmma_intrin")
    print(sch.mod["main"].script())


if __name__ == "__main__":
    test_blocked_ellmm_tensorize()
