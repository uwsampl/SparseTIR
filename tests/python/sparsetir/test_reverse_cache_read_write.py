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

@T.prim_func
def tcspmm_rev_cache_read(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 1})
    IO = T.dense_fixed(128, idtype="int32")
    JO = T.dense_variable(IO, (128, 1024), indptr, idtype="int32")
    II = T.dense_fixed(16, idtype="int32")
    JI = T.sparse_fixed(JO, (2048, 16), indices, idtype="int32", sorted=True)
    JI_dense = T.dense_fixed(16, idtype="int32")
    J = T.dense_fixed(2048, idtype="int32")
    F = T.dense_fixed(64, idtype="int32")
    A = T.match_sparse_buffer(a, [IO, JO, II, JI], dtype="float16")
    B = T.match_sparse_buffer(b, [J, F], dtype="float16")
    C = T.match_sparse_buffer(c, [IO, II, F], dtype="float16")
    JO_indptr = T.match_sparse_buffer(indptr, [IO], dtype="int32", extra_storage=1)
    JI_indices = T.match_sparse_buffer(indices, [IO, JO, JI_dense], dtype="int32")
    # body
    # with T.block("root")
    T.assume_buffer_domain(JO_indptr, [0, 1024])
    T.assume_buffer_domain(JI_indices, [0, 2048])
    for io in T.serial(128):
        with T.block("tcspmm0"):
            vio = T.axis.spatial(128, io)
            T.reads(JO_indptr[vio : vio + 2], A[vio, 0 : 128, 0 : 16, 0 : 16], B[0 : 2048, 0 : 64], JI_indices[vio, 0 : 128, 0 : 16])
            T.writes(C[vio, 0 : 16, 0 : 64])
            T.block_attr({"sparse":True})
            for jo, f_0 in T.grid(JO_indptr[vio + 1] - JO_indptr[vio], 4):
                with T.block("tcspmm1_o"):
                    vjo = T.axis.reduce(128, jo)
                    vii_o = T.axis.spatial(1, 0)
                    vji_o = T.axis.reduce(1, 0)
                    vf_o = T.axis.spatial(4, f_0)
                    T.reads(A[vio, vjo, 0 : 16, 0 : 16], B[0 : 2048, vf_o * 16 : vf_o * 16 + 16], JI_indices[vio, vjo, 0 : 16])
                    T.writes(C[vio, 0 : 16, vf_o * 16 : vf_o * 16 + 16])
                    B_shared = T.alloc_buffer([16, 16], dtype="float16", scope="shared")
                    B_shared_wmma_matrix_b = T.alloc_buffer([16, 16], dtype="float16", scope="wmma.matrix_b")
                    with T.init():
                        for ii, f_1 in T.grid(16, 16):
                            with T.block("tcspmm1_init"):
                                vii_init, vf_init = T.axis.remap("SS", [ii, f_1])
                                T.reads()
                                T.writes(C[vio, vii_init, vf_o * 16 + vf_init])
                                C[vio, vii_init, vf_o * 16 + vf_init] = T.float16(0)
                    for ax0, ax1 in T.grid(16, 16):
                        with T.block("B_shared"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            T.reads(B[JI_indices[vio, vjo, v0], vf_o * 16 + v1])
                            T.writes(B_shared[v0, v1])
                            T.block_attr({"sparse":True})
                            B_shared[v0, v1] = B[JI_indices[vio, vjo, v0], vf_o * 16 + v1]
                    for ax0, ax1 in T.grid(16, 16):
                        with T.block("B_shared_wmma.matrix_b"):
                            v0, v1 = T.axis.remap("SS", [ax0, ax1])
                            T.reads(B_shared[v0, v1])
                            T.writes(B_shared_wmma_matrix_b[v0, v1])
                            T.block_attr({"sparse":True})
                            B_shared_wmma_matrix_b[v0, v1] = B_shared[v0, v1]
                    for ii, ji, f_1 in T.grid(16, 16, 16):
                        with T.block("tcspmm1"):
                            vii, vji, vf = T.axis.remap("SRS", [ii, ji, f_1])
                            T.reads(C[vio, vii, vf_o * 16 + vf], A[vio, vjo, vii, vji], B_shared_wmma_matrix_b[vji, vf], JI_indices[vio, vjo, vji])
                            T.writes(C[vio, vii, vf_o * 16 + vf])
                            T.block_attr({"sparse":True})
                            C[vio, vii, vf_o * 16 + vf] = C[vio, vii, vf_o * 16 + vf] + A[vio, vjo, vii, vji] * B_shared_wmma_matrix_b[vji, vf]

@T.prim_func
def scatter_add(
    a: T.handle,
    b: T.handle,
    idx: T.handle,
):
    A = T.match_buffer(a, (128, 16), "float32")
    B = T.match_buffer(b, (1024,), "float32")
    I = T.match_buffer(idx, (128,), "int32")

    for i, j in T.grid(128, 16):
        with T.block("scatter"):
            vi, vj = T.axis.remap("SR", [i, j])
            with T.init():
                B[I[vi]] = T.float32(0)
            B[I[vi]] = B[I[vi]] + A[vi, vj]

@T.prim_func
def scatter_add_rev_cache_write(A: T.Buffer[(128, 16), "float32"], B: T.Buffer[(1024,), "float32"], I: T.Buffer[(128,), "int32"]) -> None:
    # body
    # with T.block("root")
    B_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
    for i, j in T.grid(128, 16):
        with T.block("scatter"):
            vi, vj = T.axis.remap("SR", [i, j])
            T.reads(I[vi], A[vi, vj])
            T.writes(B_shared[vi])
            with T.init():
                B_shared[vi] = T.float32(0)
            B_shared[vi] = B_shared[vi] + A[vi, vj]
    for ax0 in T.serial(128):
        with T.block("B_shared"):
            v0 = T.axis.spatial(128, ax0)
            T.reads(B_shared[v0])
            T.writes(B[I[v0]])
            B[I[v0]] = B_shared[v0] 


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
    B_shared = sch.reverse_cache_read(blk_inner, 2, "shared")
    B_warp = sch.reverse_cache_read(blk_inner, 2, "wmma.matrix_b")
    tvm.ir.assert_structural_equal(sch.mod["main"], tcspmm_rev_cache_read, True)


def test_rev_cache_write():
    mod = tvm.IRModule.from_expr(scatter_add)    
    sch = tir.Schedule(mod)
    blk = sch.get_block("scatter")
    sch.reverse_cache_write(blk, 0, "shared")
    tvm.ir.assert_structural_equal(sch.mod["main"], scatter_add_rev_cache_write, True)


if __name__ == "__main__":
    test_tc_spmm_cache_read()
    test_rev_cache_write()