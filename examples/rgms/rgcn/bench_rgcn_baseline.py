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
import dgl
import scipy.sparse as sp
import numpy as np
import torch as th
from torch_scatter import scatter
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from ogb.linkproppred import DglLinkPropPredDataset
from torch.profiler import profile, ProfilerActivity, schedule
from utils import get_hetero_dataset


def prepare_hetero_graph_simplified(g: dgl.DGLHeteroGraph):
    ntype_pointer = np.cumsum([0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return {
        "ntype_node_pointer": th.IntTensor(ntype_pointer).cuda(),
        "etype_edge_pointer": th.IntTensor(etype_pointer).cuda(),
    }


def test_rgcn_baseline(g: DGLHeteroGraph, feat_size: int):
    g = g.to(0)
    feat = th.rand(g.num_src_nodes(), feat_size).to(0) / 100
    out = th.zeros(g.num_dst_nodes(), feat_size).to(0) / 100
    weight = th.rand(g.num_rels, feat_size, feat_size).to(0)
    indptr, indices, eid = g.adj_sparse(fmt="csc")
    etype = g.edata[dgl.ETYPE][eid.long()]

    # dgl-lowmem
    try:
        g.srcdata["feat"] = feat.unsqueeze(-1)
        us, vs = g.edges()
        us = us.long()
        vs = vs.long()
        feat_transformed = feat[us.long()]
        msg = th.zeros(g.num_edges(), feat_size).to(0)
        weight_T = weight.permute(0, 2, 1).contiguous()
        with profile(
            activities=[ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=10, active=100),
            record_shapes=True,
        ) as prof:
            with th.no_grad():
                for epoch in range(100):
                    for i in range(1, len(g.etype_pointer)):
                        start = g.etype_pointer[i - 1]
                        end = g.etype_pointer[i]
                        msg[start:end] = feat_transformed[start:end] @ weight_T[i - 1]
                    # y_dgl_lowmem = dgl.ops.copy_e_sum(g, msg)
                    y_dgl_lowmem = scatter(msg, vs, dim=0, reduce="sum")
                    prof.step()
        measure = sum([e.cuda_time for e in prof.events()]) / 1000 / 90
        print("dgl-lowmem:\t\t {:.3f}ms".format(measure))

    except RuntimeError as err:
        print("dgl-lowmem: OOM")
        y_dgl_lowmem = None
    except BaseException as err:
        print(err)
        raise


if __name__ == "__main__":
    for feat_size in [32]:  # [4, 8, 16, 32]:
        for name in ["aifb", "mutag", "bgs", "am", "biokg"]:
            print("dataset {}, feat_size={}:".format(name, feat_size))
            dataset = get_hetero_dataset(name)
            g = dataset[0]
            type_pointers = prepare_hetero_graph_simplified(g)
            g = dgl.to_homogeneous(g)
            g.ntype_pointer = type_pointers["ntype_node_pointer"]
            g.etype_pointer = type_pointers["etype_edge_pointer"]
            g.num_ntypes = max(g.ndata[dgl.NTYPE]).item() + 1
            g.num_rels = max(g.edata[dgl.ETYPE]).item() + 1
            g = g.int()
            test_rgcn_baseline(g, feat_size)
