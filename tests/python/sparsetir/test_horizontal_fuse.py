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
    T.func_attr({"horizontal_fuse": "sequential"})
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
               
# from tvm.script import tir as T
@T.prim_func
def local_alloc(A: T.Buffer[(200,), "float32"], B: T.Buffer[(200,), "float32"]) -> None:
    # var definition
    blockIdx_x = T.env_thread("blockIdx.x")
    # body
    T.launch_thread(blockIdx_x, 200)
    # if blockIdx_x < 100:
    C_local = T.allocate([1], "float32", "local")
    C_local[0] = T.float32(0)
    A[blockIdx_x] = C_local[0]
    # else:
    C_local_1 = T.allocate([1], "float32", "local")
    C_local_1[0] = T.float32(0)
    B[blockIdx_x] = C_local_1[0]


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
    print(f.imported_modules[0].get_source())

    # x_np = np.random.rand(128, 128).astype("float32")
    # y_np = np.random.rand(64, 64).astype("float32")
    # z1_np = np.zeros(128).astype("float32")
    # z2_np = np.zeros(64).astype("float32")

    # z1_golden = x_np.sum(axis=-1)
    # z2_golden = y_np.sum(axis=-1)

    # x = tvm.nd.array(x_np, device=tvm.cuda(0))
    # y = tvm.nd.array(y_np, device=tvm.cuda(0))
    # z1 = tvm.nd.array(z1_np, device=tvm.cuda(0))
    # z2 = tvm.nd.array(z2_np, device=tvm.cuda(0))

    # f(x, y, z1, z2)

    # tvm.testing.assert_allclose(z1.numpy(), z1_golden, rtol=1e-5, atol=1e-5)
    # tvm.testing.assert_allclose(z2.numpy(), z2_golden, rtol=1e-5, atol=1e-5)


def test_local_alloc():
    mod = tvm.IRModule.from_expr(local_alloc)
    mod = tir.transform.StorageRewrite()(mod)
    print(mod["main"].script())


if __name__ == "__main__":
    test_end_to_end()
    # test_local_alloc()
