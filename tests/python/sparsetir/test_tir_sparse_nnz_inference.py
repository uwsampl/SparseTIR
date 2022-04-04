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
import scipy.sparse as sp
import numpy as np
from tvm.script import tir as T

@T.prim_func
def csr2bsr_cnt_nnz(
    indptr: T.handle, indices: T.handle,
    new_cord: T.handle, glb_counter: T.handle,
    n: T.int32, m: T.int32, nnz: T.int32) -> None:
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (m, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(2)
    New_cord = T.match_sparse_buffer(new_cord, (I, J, K), "int32")
    with T.iter([I, J], "SS", "csr2bsr_cnt_nnz") as [vi, vj]:
        New_cord[vi, vj, 0] = 0
        New_cord[vi, vj, 1] = 1
        

@T.prim_func
def csr2bsr(indptr_1: T.handle, indices_1: T.handle, indptr_2: T.handle, indices_2: T.handle,
    a_csr: T.handle, a_bsr: T.handle,
    block_size: T.int32,
    n: T.int32, m: T.int32, nnz: T.int32,
    nb: T.int32, mb: T.int32, nnzb: T.int32) -> None:
    I = T.dense_fixed(n)
    J = T.sparse_variable(I, (m, nnz), (indptr_1, indices_1), "int32")
    Ibo = T.dense_fixed(nb)
    Jbo = T.sparse_variable(Ibo, (mb, nnzb), (indptr_2, indices_2), "int32")
    Ibi = T.dense_fixed(block_size)
    Jbi = T.dense_fixed(block_size)
    A_csr = T.match_sparse_buffer(a_csr, (I, J), "float32")
    A_bsr = T.match_sparse_buffer(a_bsr, (Ibo, Jbo, Ibi, Jbi), "float32")
    with T.iter([I, J], "SS", "csr2bsrm") as [vi, vj]:
        A_bsr[T.floordiv(vi, block_size), T.floordiv(vj, block_size), T.floormod(vi, block_size), T.floormod(vj, block_size)] =\
            A_csr[vi, vj]


def test_cnt_nnz():
    mod = tvm.IRModule.from_expr(csr2bsr_cnt_nnz)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    print(mod['main'].script())


def test_csr2bsr():
    mod = tvm.IRModule.from_expr(csr2bsr)
    mod = tvm.tir.transform.LowerSparseTIR()(mod)
    print(mod['main'].script())


if __name__ == "__main__":
    test_cnt_nnz()
    test_csr2bsr()