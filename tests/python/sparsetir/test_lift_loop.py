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
import sys

import pytest
import tvm
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def nested_loop_no_dependency() -> None:
    a = T.alloc_buffer((16, 16), "float32")
    b = T.alloc_buffer((16, 16), "float32")
    for i in T.grid(16):
        with T.block("outer"):
            vi = T.axis.spatial(16, i)
            for jo in T.grid(4):
                for ji in T.grid(4):
                    with T.block("inner"):
                        vj = T.axis.spatial(16, 4 * jo + ji)
                        a[vi, vj] = b[vi, vj]


@T.prim_func
def nested_loop_no_dependency_after_transform() -> None:
    a = T.alloc_buffer([16, 16], dtype="float32")
    b = T.alloc_buffer([16, 16], dtype="float32")
    for i, jo in T.grid(16, 4):
        with T.block("outer"):
            vi = T.axis.spatial(16, i)
            v_jo = T.axis.spatial(4, jo)
            T.reads(b[vi, 0 : 16])
            T.writes(a[vi, 0 : 16])
            for ji in T.serial(4):
                with T.block("inner"):
                    vj = T.axis.spatial(16, 4 * v_jo + ji)
                    T.reads(b[vi, vj])
                    T.writes(a[vi, vj])
                    a[vi, vj] = b[vi, vj]

# @T.prim_func
# def nested_loop_with_dependency() -> None:
#     pass


def test_nested_loop_no_dependency() -> None:
    sch = tvm.tir.Schedule(nested_loop_no_dependency)
    inner = sch.get_block("inner")
    jo, ji = sch.get_loops(inner)
    sch.lift_loop(jo)
    tvm.ir.assert_structural_equal(sch.mod["main"], nested_loop_no_dependency_after_transform)
    verify_trace_roundtrip(sch=sch, mod=nested_loop_no_dependency)


if __name__ == "__main__":
    test_nested_loop_no_dependency()