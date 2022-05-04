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
# pylint: disable=missing-function-docstring,missing-module-docstring

import tvm
from tvm.script import tir as T


@T.prim_func
def independent() -> None:
    A = T.alloc_buffer((32, 32), "float32")
    B = T.alloc_buffer((32), "float32")
    C = T.alloc_buffer((32, 32), "float32")
    for i in range(32):
        for j in range(32):
            with T.block("B"):
                vi = T.axis.spatial(32, i)
                B[vi] = A[vi, vi]
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi] + A[vi, vj]


@T.prim_func
def after_transform() -> None:
    # body
    # with T.block("root")
    A = T.alloc_buffer([32, 32], dtype="float32")
    B = T.alloc_buffer([32], dtype="float32")
    C = T.alloc_buffer([32, 32], dtype="float32")
    for i in T.serial(32):
        with T.block("B"):
            vi = T.axis.spatial(32, i)
            T.reads(A[vi, vi])
            T.writes(B[vi])
            B[vi] = A[vi, vi]
        for j in T.serial(32):
            T.evaluate(0)
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi], A[vi, vj])
                T.writes(C[vi, vj])
                C[vi, vj] = B[vi] + A[vi, vj]


def test_place_under():
    sch = tvm.tir.Schedule(independent, debug_mask="all")
    block_B = sch.get_block("B")
    i, j = sch.get_loops(block_B)
    sch.place_under(block_B, i)
    tvm.ir.assert_structural_equal(sch.mod["main"], after_transform)


if __name__ == "__main__":
    test_place_under()
