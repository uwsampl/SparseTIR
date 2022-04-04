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


from dgl.heterograph import DGLHeteroGraph
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import dgl
import dgl.function as fn
import torch as th
from tvm.script import tir as T
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from sparse_tir_scripts import rgcn_hetero_forward
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer
from typing import List, Tuple


def get_dataset_by_name(name: str):
    if name == "aifb":
        return AIFBDataset()
    elif name == "mutag":
        return MUTAGDataset()
    elif name == "bgs":
        return BGSDataset()
    elif name == "am":
        return AMDataset()
    else:
        raise KeyError("Unknown dataset {}.".format(name))


class TorchOpTimer(object):
    def __enter__(self):
        self.start_event = th.cuda.Event(enable_timing=True)
        self.end_event = th.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        th.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event)


def prepare_hetero_graph_simplified(g: dgl.DGLHeteroGraph):
    ntype_pointer = np.cumsum([0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return {
        "ntype_node_pointer": th.IntTensor(ntype_pointer),
        "etype_edge_pointer": th.IntTensor(etype_pointer),
    }


blks = ["blockIdx.x", "blockIdx.y", "blockIdx.z"]

def test_lower_rgcn_hetero(
    g: dgl.DGLHeteroGraph,
    feat_size: int,
    blk_order: List[Tuple[str]],
    split_factor_f: int,
    split_factor_i: int,
):
    N, R, FEAT_SIZE, NNZ_I, NNZ_J = rgcn_hetero_forward.params[-5:]
    n = g.num_nodes()
    r = len(g.etypes)
    nnz_j = g.num_edges()

    feat = th.rand(n, feat_size).to(0) / 100
    out = th.zeros(n, feat_size).to(0) / 100
    weight = th.rand(r, feat_size, feat_size).to(0)
    W = tvm.nd.array(weight.view(-1).cpu().numpy().astype("float32"), device=tvm.cuda(0))
    X = tvm.nd.array(feat.view(-1).cpu().numpy().astype("float32"), device=tvm.cuda(0))
    Y = tvm.nd.array(out.view(-1).cpu().numpy().astype("float32"), device=tvm.cuda(0))

    indptr_i = [th.LongTensor([0])]
    indices_i = []
    indptr_j = [th.LongTensor([0])]
    indices_j = []
    for etype in g.canonical_etypes:
        src_type, _, dst_type = etype
        etype_id = g.get_etype_id(etype)
        src_type_id = g.get_ntype_id(src_type)
        dst_type_id = g.get_ntype_id(dst_type)
        g_sub = g[etype]
        indptr, indices, _ = g_sub.adj_sparse(fmt="csc")

        unique_nodes = th.nonzero(indptr[:-1] != indptr[1:]).squeeze(1)
        indptr_i.append(th.LongTensor([len(unique_nodes)]))
        indices_i.append(unique_nodes + g.ntype_pointer[dst_type_id])
        indptr_j.append(indptr[unique_nodes] + g.etype_pointer[etype_id])
        indices_j.append(indices + g.ntype_pointer[src_type_id])

    indptr_i = tvm.nd.array(th.cat(indptr_i).numpy().astype("int32"), device=tvm.cuda(0))
    indices_i = tvm.nd.array(th.cat(indices_i).numpy().astype("int32"), device=tvm.cuda(0))
    indptr_j = tvm.nd.array(th.cat(indptr_j).numpy().astype("int32"), device=tvm.cuda(0))
    indices_j = tvm.nd.array(th.cat(indices_j).numpy().astype("int32"), device=tvm.cuda(0))

    nnz_i = indices_i.shape[0]
    mod = tvm.IRModule.from_expr(
        rgcn_hetero_forward.specialize(
            {N: n, R: r, FEAT_SIZE: feat_size, NNZ_I: nnz_i, NNZ_J: nnz_j}
        )
    )
    mod = lower_sparse_iter(mod)
    sch = tir.Schedule(mod)

    blk0 = sch.get_block("rgcn-hetero-forward0")
    blk1 = sch.get_block("rgcn-hetero-forward1")
    blk2 = sch.get_block("rgcn-hetero-forward2")
    read_blk = sch.cache_read(blk1, 2, "shared")
    write_blk = sch.cache_write(blk2, 0, "local")
    sch.annotate(write_blk, "atomic", True)
    f_out, r = sch.get_loops(blk0)
    f_out_o, f_out_i = sch.split(f_out, [split_factor_f, None])
    (i,) = sch.get_loops(blk1)
    j, f_in = sch.get_loops(blk2)
    sch.reorder(f_in, j)
    i1, i2 = sch.split(i, [None, split_factor_i])
    sch.lift_loop(i2)
    sch.bind(i2, blks[blk_order[0]])
    sch.bind(r, blks[blk_order[1]])
    sch.bind(f_out_o, blks[blk_order[2]])
    sch.bind(f_in, "threadIdx.x")
    sch.bind(f_out_i, "threadIdx.y")
    _, _, ax2 = sch.get_loops(read_blk)
    sch.bind(ax2, "threadIdx.x")
    mod = lower_sparse_buffer(sch.mod)
    print(mod["main"].script())

    f = tvm.build(mod["main"], target="cuda")
    print(f.imported_modules[0].get_source())

    # cold_start = 3
    # total = 10
    # accum = 0

    # for epoch in range(10):
    #     with TorchOpTimer() as timer:
    #         f(W, X, Y, indptr_i, indices_i, indptr_j, indices_j)
    #     if epoch >= cold_start:
    #         accum += timer.time

    # print("sparse-tir:\t\t {}ms".format(accum / (total - cold_start)))


if __name__ == "__main__":
    for feat_size in [32]:  # [4, 8, 16, 32, 64]:
        for name in ["am"]:  # ['aifb', 'mutag', 'bgs', 'am']:
            dataset = get_dataset_by_name(name)
            g = dataset[0]
            type_pointers = prepare_hetero_graph_simplified(g)
            g.ntype_pointer = type_pointers["ntype_node_pointer"]
            g.etype_pointer = type_pointers["etype_edge_pointer"]
            for blk_order in [(2, 0, 1)]:
                for split_factor_f in [8]:
                    for split_factor_i in [512]:
                        print(
                            "dataset {}, blk_order {}, split_factor_f {}, split_factor_i {}:".format(
                                name, blk_order, split_factor_f, split_factor_i
                            )
                        )
                        test_lower_rgcn_hetero(
                            g, feat_size, blk_order, split_factor_f, split_factor_i
                        )
