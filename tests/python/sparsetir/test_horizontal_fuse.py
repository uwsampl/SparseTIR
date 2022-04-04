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
    B: T.Buffer[(64, 64), "float32"],
    C1: T.Buffer[(128,), "float32"],
    C2: T.Buffer[(64,), "float32"],
) -> None:
    T.func_attr({"horizontal_fuse": True})
    for i, j in T.grid(128, 128):
        with T.block("first"):
            vi, vj = T.axis.remap("SR", [i, j])
            with T.init():
                C1[vi] = T.float32(0)
            C1[vi] = C1[vi] + A[vi, vj]
    for i, j in T.grid(64, 64):
        with T.block("second"):
            vi, vj = T.axis.remap("SR", [i, j])
            with T.init():
                C2[vi] = T.float32(0)
            C2[vi] = C2[vi] + B[vi, vj]


@T.prim_func
def before_fuse(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(64, 64), "float32"],
    C: T.Buffer[(192,), "float32"],
) -> None:
    T.func_attr({"horizontal_fuse": True})
    for i in T.thread_binding(0, 128, "blockIdx.x"):
        for j in T.thread_binding(0, 128, "threadIdx.x"):
            with T.block():
                C[i] = C[i] + A[i, j]
    for i in T.thread_binding(0, 64, "blockIdx.x"):
        for j in T.thread_binding(0, 64, "threadIdx.x"):
            with T.block():
                C[i + 128] = C[i + 128] + B[i, j]


@T.prim_func
def after_fuse(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(64, 64), "float32"],
    C: T.Buffer[(192,), "float32"],
) -> None:
    for block_idx in T.thread_binding(192, thread="blockIdx.x"):
        for j in T.thread_binding(128, thread="threadIdx.x"):
            with T.block():
                T.where(block_idx < 128)
                T.reads(C[block_idx], A[block_idx, j])
                T.writes(C[block_idx])
                C[block_idx] = C[block_idx] + A[block_idx, j]
        for j in T.thread_binding(128, thread="threadIdx.x"):
            with T.block():
                T.where(128 <= block_idx and j < 64)
                T.reads(C[block_idx], B[block_idx - 128, j])
                T.writes(C[block_idx])
                C[block_idx] = C[block_idx] + B[block_idx - 128, j]


@T.prim_func
def before_fuse_with_other_func_attr(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(64, 64), "float32"],
    C: T.Buffer[(192,), "float32"],
) -> None:
    T.func_attr({"horizontal_fuse": True, "placeholder": 1})
    for i in T.thread_binding(0, 128, "blockIdx.x"):
        for j in T.thread_binding(0, 128, "threadIdx.x"):
            with T.block():
                C[i] = C[i] + A[i, j]
    for i in T.thread_binding(0, 64, "blockIdx.x"):
        for j in T.thread_binding(0, 64, "threadIdx.x"):
            with T.block():
                C[i + 128] = C[i + 128] + B[i, j]


@T.prim_func
def after_fuse_with_other_func_attr(
    A: T.Buffer[(128, 128), "float32"],
    B: T.Buffer[(64, 64), "float32"],
    C: T.Buffer[(192,), "float32"],
) -> None:
    T.func_attr({"placeholder": 1})
    for block_idx in T.thread_binding(192, thread="blockIdx.x"):
        for j in T.thread_binding(128, thread="threadIdx.x"):
            with T.block():
                T.where(block_idx < 128)
                T.reads(C[block_idx], A[block_idx, j])
                T.writes(C[block_idx])
                C[block_idx] = C[block_idx] + A[block_idx, j]
        for j in T.thread_binding(128, thread="threadIdx.x"):
            with T.block():
                T.where(128 <= block_idx and j < 64)
                T.reads(C[block_idx], B[block_idx - 128, j])
                T.writes(C[block_idx])
                C[block_idx] = C[block_idx] + B[block_idx - 128, j]


@T.prim_func
def fix_storage_rewrite(A: T.Buffer[(16384,), "float32"], B: T.Buffer[(4096,), "float32"], C1: T.Buffer[(128,), "float32"], C2: T.Buffer[(64,), "float32"]) -> None:
    T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
    # var definition
    threadIdx_x = T.env_thread("threadIdx.x")
    blockIdx_x = T.env_thread("blockIdx.x")
    T.preflattened_buffer(A, [128, 128], dtype="float32", data=A.data)
    T.preflattened_buffer(B, [64, 64], dtype="float32", data=B.data)
    T.preflattened_buffer(C1, [128], dtype="float32", data=C1.data)
    T.preflattened_buffer(C2, [64], dtype="float32", data=C2.data)
    # body
    T.launch_thread(blockIdx_x, 192)
    T.launch_thread(threadIdx_x, 128)
    cross_thread_0 = T.allocate([1], "float32", "local")
    if blockIdx_x < 128:
        with T.attr(T.comm_reducer(lambda x, y: x + y, [T.float32(0)]), "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle")):
            T.evaluate(T.tvm_thread_allreduce(T.uint32(1), A[blockIdx_x * 128 + threadIdx_x], True, cross_thread_0[0], threadIdx_x, dtype="handle"))
        C1[blockIdx_x] = cross_thread_0[0]
    if 128 <= blockIdx_x and threadIdx_x < 64:
        with T.attr(T.comm_reducer(lambda x_1, y_1: x_1 + y_1, [T.float32(0)]), "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle")):
            T.evaluate(T.tvm_thread_allreduce(T.uint32(1), B[blockIdx_x * 64 + threadIdx_x - 8192], True, cross_thread_0[0], threadIdx_x, dtype="handle"))
        C2[blockIdx_x - 128] = cross_thread_0[0]


def test_end_to_end():
    sch = tvm.tir.Schedule(original)
    blk1 = sch.get_block("first")
    blk2 = sch.get_block("second")
    i, j = sch.get_loops(blk1)
    sch.bind(i, "blockIdx.x")
    sch.bind(j, "threadIdx.x")
    i, j = sch.get_loops(blk2)
    sch.bind(i, "blockIdx.x")
    sch.bind(j, "threadIdx.x")
    f = tvm.build(sch.mod["main"], target="cuda")

    x_np = np.random.rand(128, 128).astype("float32")
    y_np = np.random.rand(64, 64).astype("float32")
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


def test_horizontal_fuse():
    mod = tvm.IRModule.from_expr(before_fuse)
    mod = tvm.tir.transform.HorizontalFusion()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], after_fuse)


def test_horizontal_fuse_with_other_func_attr():
    mod = tvm.IRModule.from_expr(before_fuse_with_other_func_attr)
    mod = tvm.tir.transform.HorizontalFusion()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], after_fuse_with_other_func_attr)


def test_fix_storage_rewrite():
    mod = tvm.IRModule.from_expr(fix_storage_rewrite)
    mod = tvm.build(mod, target="cuda")
    # mod = tvm.tir.transform.LowerThreadAllreduce()(mod)
    # print(mod.imported_modules[0].get_source())


if __name__ == "__main__":
    test_end_to_end()
    test_horizontal_fuse()
    test_horizontal_fuse_with_other_func_attr()
    test_fix_storage_rewrite()
