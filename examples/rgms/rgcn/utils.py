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

import dgl
import numpy as np
import torch as th
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from ogb.linkproppred import DglLinkPropPredDataset
from tvm.script import tir as T


def get_hetero_dataset(name: str):
    if name == "aifb":
        return AIFBDataset()
    elif name == "mutag":
        return MUTAGDataset()
    elif name == "bgs":
        return BGSDataset()
    elif name == "am":
        return AMDataset()
    elif name == "biokg":
        return DglLinkPropPredDataset(name="ogbl-biokg")
    else:
        raise KeyError("Unknown dataset {}.".format(name))


def get_type_pointers(g: dgl.DGLHeteroGraph):
    ntype_pointer = np.cumsum([0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return {
        "ntype_node_pointer": th.IntTensor(ntype_pointer),
        "etype_edge_pointer": th.IntTensor(etype_pointer),
    }


@T.prim_func
def rgcn_hetero_forward(
    a: T.handle,
    w: T.handle,
    x: T.handle,
    y: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rels: T.int32,
    feat_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    R = T.dense_fixed(num_rels)
    I = T.sparse_variable(R, (m, nnz_i), (indptr_i, indices_i), "int32")
    J = T.sparse_variable(I, (n, nnz_j), (indptr_j, indices_j), "int32")
    I_detach = T.dense_fixed(m)
    J_detach = T.dense_fixed(n)
    F_in = T.dense_fixed(feat_size)
    F_out = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (R, I, J), "float32")
    W = T.match_sparse_buffer(w, (R, F_out, F_in), "float32")
    X = T.match_sparse_buffer(x, (J_detach, F_in), "float32")
    Y = T.match_sparse_buffer(y, (I_detach, F_out), "float32")
    with T.iter([F_out, R, I, J, F_in], "SSSRR", "rgcn-hetero-forward") as [fo, r, i, j, fi]:
        with T.init():
            Y[i, fo] = 0.0
        Y[i, fo] = Y[i, fo] + A[r, i, j] * W[r, fo, fi] * X[j, fi]


@T.prim_func
def ell3d(
    a: T.handle,
    indptr_io: T.handle,
    indices_ii: T.handle,
    indices_j: T.handle,
    d0: T.int32,
    d1: T.int32,
    d2: T.int32,
    nnz: T.int32,
    nnz_rows: T.int32,
    nnz_cols: T.int32,
) -> None:
    R = T.dense_fixed(d0, idtype="int32")
    IO = T.dense_variable(R, (d1, nnz), indptr_io, idtype="int32")
    II = T.sparse_fixed(IO, (d2, nnz_rows), indices_ii, idtype="int32")
    J = T.sparse_fixed(II, (d2, nnz_cols), indices_j, idtype="int32")
    A = T.match_sparse_buffer(a, (R, IO, II, J), dtype="float32")
    T.evaluate(0)
