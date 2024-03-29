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
import tvm.testing
from tvm import tir
from tvm.script import tir as T


@T.prim_func
def original(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(64, 128), "float32"],
    C1: T.Buffer[(128,), "float32"],
    C2: T.Buffer[(64,), "float32"],
) -> None:
    T.func_attr({"horizontal_fuse": 1})
    for i, j in T.grid(128, 128):
        with T.block("first"):
            vi, vj = T.axis.remap("SR", [i, j])
            with T.init():
                C1[vi] = T.float32(0)
            C1[vi] = C1[vi] + A[vi, vj]
    for i, j in T.grid(64, 128):
        with T.block("second"):
            vi, vj = T.axis.remap("SR", [i, j])
            with T.init():
                C2[vi] = T.float32(0)
            C2[vi] = C2[vi] + B[vi, vj]


@T.prim_func
def before_horizontal_fuse(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(64, 128), "float32"],
    C1: T.Buffer[(128,), "float32"],
    C2: T.Buffer[(64,), "float32"],
) -> None:
    # function attr dict
    T.func_attr({"horizontal_fuse": 1})
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
    B_shared = T.alloc_buffer([64, 128], dtype="float32", scope="shared")
    C1_local = T.alloc_buffer([128], dtype="float32", scope="local")
    C2_local = T.alloc_buffer([64], dtype="float32", scope="local")
    for i_0 in T.thread_binding(32, thread="blockIdx.x"):
        for i_1 in T.serial(4):
            for ax0 in T.serial(128):
                with T.block("A_shared"):
                    v0 = T.axis.spatial(128, i_0 * 4 + i_1)
                    v1 = T.axis.spatial(128, ax0)
                    T.reads(A[v0, v1])
                    T.writes(A_shared[v0, v1])
                    A_shared[v0, v1] = A[v0, v1]
            for j in T.thread_binding(128, thread="threadIdx.x"):
                with T.block("first"):
                    vi = T.axis.spatial(128, i_0 * 4 + i_1)
                    vj = T.axis.reduce(128, j)
                    T.reads(A_shared[vi, vj])
                    T.writes(C1_local[vi])
                    with T.init():
                        C1_local[vi] = T.float32(0)
                    C1_local[vi] = C1_local[vi] + A_shared[vi, vj]
            with T.block("C1_local"):
                v0 = T.axis.spatial(128, i_0 * 4 + i_1)
                T.reads(C1_local[v0])
                T.writes(C1[v0])
                C1[v0] = C1_local[v0]
    for i_0 in T.thread_binding(16, thread="blockIdx.x"):
        for i_1 in T.serial(4):
            for ax0 in T.serial(128):
                with T.block("B_shared"):
                    v0 = T.axis.spatial(64, i_0 * 4 + i_1)
                    v1 = T.axis.spatial(128, ax0)
                    T.reads(B[v0, v1])
                    T.writes(B_shared[v0, v1])
                    B_shared[v0, v1] = B[v0, v1]
            for j in T.thread_binding(128, thread="threadIdx.x"):
                with T.block("second"):
                    vi = T.axis.spatial(64, i_0 * 4 + i_1)
                    vj = T.axis.reduce(128, j)
                    T.reads(B_shared[vi, vj])
                    T.writes(C2_local[vi])
                    with T.init():
                        C2_local[vi] = T.float32(0)
                    C2_local[vi] = C2_local[vi] + B_shared[vi, vj]
            with T.block("C2_local"):
                v0 = T.axis.spatial(64, i_0 * 4 + i_1)
                T.reads(C2_local[v0])
                T.writes(C2[v0])
                C2[v0] = C2_local[v0]


@T.prim_func
def after_horizontal_fuse(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(64, 128), "float32"],
    C1: T.Buffer[(128,), "float32"],
    C2: T.Buffer[(64,), "float32"],
) -> None:
    # body
    # with T.block("root")
    A_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
    B_shared = T.alloc_buffer([64, 128], dtype="float32", scope="shared")
    C1_local = T.alloc_buffer([128], dtype="float32", scope="local")
    C2_local = T.alloc_buffer([64], dtype="float32", scope="local")
    for block_idx_x in T.thread_binding(48, thread="blockIdx.x"):
        for thread_idx_x in T.thread_binding(128, thread="threadIdx.x"):
            if block_idx_x < 32:
                for i_1 in T.serial(4):
                    for ax0 in T.serial(128):
                        with T.block("A_shared"):
                            v0 = T.axis.spatial(128, block_idx_x * 4 + i_1)
                            v1 = T.axis.spatial(128, ax0)
                            T.reads(A[v0, v1])
                            T.writes(A_shared[v0, v1])
                            A_shared[v0, v1] = A[v0, v1]
                    with T.block("first"):
                        vi = T.axis.spatial(128, block_idx_x * 4 + i_1)
                        vj = T.axis.reduce(128, thread_idx_x)
                        T.reads(A_shared[vi, vj])
                        T.writes(C1_local[vi])
                        with T.init():
                            C1_local[vi] = T.float32(0)
                        C1_local[vi] = C1_local[vi] + A_shared[vi, vj]
                    with T.block("C1_local"):
                        v0 = T.axis.spatial(128, block_idx_x * 4 + i_1)
                        T.reads(C1_local[v0])
                        T.writes(C1[v0])
                        C1[v0] = C1_local[v0]
            else:
                if block_idx_x < 48:
                    for i_1 in T.serial(4):
                        for ax0 in T.serial(128):
                            with T.block("B_shared"):
                                v0 = T.axis.spatial(64, (block_idx_x - 32) * 4 + i_1)
                                v1 = T.axis.spatial(128, ax0)
                                T.reads(B[v0, v1])
                                T.writes(B_shared[v0, v1])
                                B_shared[v0, v1] = B[v0, v1]
                        with T.block("second"):
                            vi = T.axis.spatial(64, (block_idx_x - 32) * 4 + i_1)
                            vj = T.axis.reduce(128, thread_idx_x)
                            T.reads(B_shared[vi, vj])
                            T.writes(C2_local[vi])
                            with T.init():
                                C2_local[vi] = T.float32(0)
                            C2_local[vi] = C2_local[vi] + B_shared[vi, vj]
                        with T.block("C2_local"):
                            v0 = T.axis.spatial(64, (block_idx_x - 32) * 4 + i_1)
                            T.reads(C2_local[v0])
                            T.writes(C2[v0])
                            C2[v0] = C2_local[v0]


def test_horizontal_fuse_pass():
    mod = tvm.IRModule.from_expr(before_horizontal_fuse)
    mod = tvm.tir.transform.HorizontalFusion()(mod)
    tvm.ir.assert_structural_equal(mod["main"], after_horizontal_fuse)


def test_end_to_end():
    sch = tvm.tir.Schedule(original)
    blk1 = sch.get_block("first")
    blk2 = sch.get_block("second")
    A_read = sch.cache_read(blk1, 0, "shared")
    B_read = sch.cache_read(blk2, 0, "shared")
    C_write_0 = sch.cache_write(blk1, 0, "local")
    C_write_1 = sch.cache_write(blk2, 0, "local")
    i, j = sch.get_loops(blk1)
    sch.compute_at(A_read, i)
    sch.reverse_compute_at(C_write_0, i)
    io, ii = sch.split(i, [None, 4])
    sch.bind(io, "blockIdx.x")
    sch.bind(j, "threadIdx.x")
    i, j = sch.get_loops(blk2)
    sch.compute_at(B_read, i)
    sch.reverse_compute_at(C_write_1, i)
    io, ii = sch.split(i, [None, 4])
    sch.bind(io, "blockIdx.x")
    sch.bind(j, "threadIdx.x")
    f = tvm.build(sch.mod["main"], target="cuda")

    x_np = np.random.rand(128, 128).astype("float32")
    y_np = np.random.rand(64, 128).astype("float32")
    z1_np = np.zeros(128).astype("float32")
    z2_np = np.zeros(64).astype("float32")

    z1_golden = x_np.sum(axis=-1)
    z2_golden = y_np.sum(axis=-1)

    x = tvm.nd.array(x_np, device=tvm.cuda(0))
    y = tvm.nd.array(y_np, device=tvm.cuda(0))
    z1 = tvm.nd.array(z1_np, device=tvm.cuda(0))
    z2 = tvm.nd.array(z2_np, device=tvm.cuda(0))

    f(x, y, z1, z2)

    tvm.testing.assert_allclose(z1.numpy(), z1_golden, rtol=1e-5, atol=1e-5)
    tvm.testing.assert_allclose(z2.numpy(), z2_golden, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_end_to_end()
    test_horizontal_fuse_pass()
