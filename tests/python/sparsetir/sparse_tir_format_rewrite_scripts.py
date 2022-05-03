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


@T.prim_func
def bsr(
    a: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
    block_size: T.int32,
) -> None:
    IO = T.dense_fixed(m)
    JO = T.sparse_variable(IO, (n, nnz), (indptr, indices), "int32")
    II = T.dense_fixed(block_size)
    JI = T.dense_fixed(block_size)
    A = T.match_sparse_buffer(a, (IO, JO, II, JI), "float32")
    T.evaluate(0)


@T.prim_func
def after_bsr_32_rewrite(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    k: T.int32,
    nnz: T.int32,
    a_bsr: T.handle,
    indptr_bsr: T.handle,
    indices_bsr: T.handle,
    m_bsr: T.int32,
    n_bsr: T.int32,
    nnz_bsr: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K), "float32")
    C = T.match_sparse_buffer(c, (I, K), "float32")
    IO = T.dense_fixed(m_bsr)
    JO = T.sparse_variable(IO, (n_bsr, nnz_bsr), (indptr_bsr, indices_bsr), "int32")
    II = T.dense_fixed(32)
    JI = T.dense_fixed(32)
    A_bsr = T.match_sparse_buffer(a_bsr, (IO, JO, II, JI), "float32")
    with T.iter([IO, II, K, JO, JI], "SSR", "csrmm") as [vio, vii, vk, vjo, vji]:
        with T.init():
            C[vio * 32 + vii, vk] = 0.0
        C[vio * 32 + vii, vk] = (
            C[vio * 32 + vii, vk] + A_bsr[vio, vjo, vii, vji] * B[vjo * 32 + vji, vk]
        )
