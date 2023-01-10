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

import enum
import dgl
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import dgl.function as fn
import torch as th
from tvm.script import tir as T
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from tvm.sparse import (
    lower_sparse_iter,
    lower_sparse_buffer,
    FormatRewriteRule,
    format_decompose,
    csf_to_ell3d,
)
from typing import List, Tuple, Mapping
from utils import get_hetero_dataset, rgcn_hetero_forward, ell3d, get_type_pointers


def convert_indptr_to_mid_array(indptr):
    indptr_numpy = indptr.numpy()
    ret = []
    for i in range(len(indptr_numpy) - 1):
        ret.append(np.zeros((indptr_numpy[i + 1] - indptr_numpy[i],), dtype=np.int32) + i)
    return np.concatenate(ret, axis=-1)


def get_ground_truth(
    g: dgl.DGLHeteroGraph,
    type_pointers: Mapping[str, th.Tensor],
    feat: th.Tensor,
    weight: th.Tensor,
) -> th.Tensor:
    feat_size = feat.shape[-1]
    g_homo = dgl.to_homogeneous(g)
    g_homo = g_homo.to(0)
    weight_T = weight.permute(0, 2, 1).contiguous()
    etype_pointer = type_pointers["etype_edge_pointer"]
    try:
        g_homo.srcdata["feat"] = feat.unsqueeze(-1)
        us, vs = g_homo.edges()
        feat_transformed = feat[us]
        msg = th.zeros(g_homo.num_edges(), feat_size).to(0)
        with th.no_grad():
            for i in range(1, len(etype_pointer)):
                start = etype_pointer[i - 1]
                end = etype_pointer[i]
                msg[start:end] = feat_transformed[start:end] @ weight_T[i - 1]
            y_dgl_lowmem = dgl.ops.copy_e_sum(g_homo, msg)
    except RuntimeError as err:
        print("dgl-lowmem: OOM")
        y_dgl_lowmem = None
    return y_dgl_lowmem


def csf_to_ell3d_inv_idx_map(r, io, ii, j):
    return r, ii, j


def csf_to_ell3d_idx_map(r, i, j):
    return r, 0, i, j


def test_rgcn_composable_format(
    g: dgl.DGLHeteroGraph,
    type_pointers: Mapping[str, th.Tensor],
    feat_size: int,
    feat: th.Tensor,
    weight: th.Tensor,
    ground_truth: th.Tensor,
    split_factor_f: int,
    group_size: 32,
    buckets: List[int] = [1, 2, 4, 8],
):
    # preprocess data
    ntype_node_pointer = type_pointers["ntype_node_pointer"]
    etype_edge_pointer = type_pointers["etype_edge_pointer"]
    csf_indptr_0 = [0]
    csf_indices_0 = []
    csf_indptr_1 = [th.tensor([0], dtype=th.int32)]
    csf_indices_1 = []
    num_rels = len(g.canonical_etypes)
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    for etype in g.canonical_etypes:
        src_type, _, dst_type = etype
        etype_id = g.get_etype_id(etype)
        src_type_id = g.get_ntype_id(src_type)
        dst_type_id = g.get_ntype_id(dst_type)
        g_sub = g[etype]
        m_sub, n_sub = g_sub.num_dst_nodes(), g_sub.num_src_nodes()
        indptr, indices, _ = g_sub.adj_sparse(fmt="csc")
        csf_indptr_0.append(csf_indptr_0[-1] + m_sub)
        csf_indices_0.append(ntype_node_pointer[dst_type_id] + th.arange(m_sub, dtype=th.int32))
        csf_indptr_1.append(csf_indptr_1[-1][-1] + indptr[1:])
        csf_indices_1.append(ntype_node_pointer[src_type_id] + indices)

    csf_indptr_0 = th.tensor(csf_indptr_0, dtype=th.int32)
    csf_indices_0 = th.cat(csf_indices_0, dim=-1)
    csf_indptr_1 = th.cat(csf_indptr_1, dim=-1)
    csf_indices_1 = th.cat(csf_indices_1, dim=-1)

    dev = tvm.cpu(0)
    csf_indptr_0_nd = tvm.nd.array(csf_indptr_0.int(), device=dev)
    csf_indices_0_nd = tvm.nd.array(csf_indices_0.int(), device=dev)
    csf_indptr_1_nd = tvm.nd.array(csf_indptr_1.int(), device=dev)
    csf_indices_1_nd = tvm.nd.array(csf_indices_1.int(), device=dev)
    buckets_row = [group_size // _ for _ in buckets]

    indptr, row_indices, col_indices, mask = csf_to_ell3d(
        csf_indptr_0_nd,
        csf_indices_0_nd,
        csf_indptr_1_nd,
        csf_indices_1_nd,
        buckets_row,
        buckets,
    )
    mids = list(map(convert_indptr_to_mid_array, indptr))

    d0, d1, d2, nnz, nnz_rows, nnz_cols = ell3d.params[-6:]
    rewrites = []
    for bucket_id, bucket_size in enumerate(buckets):
        rewrites.append(
            FormatRewriteRule(
                str(bucket_id),
                ell3d.specialize(
                    {
                        d0: num_rels,
                        d1: m,
                        d2: n,
                        nnz: row_indices[bucket_id].shape[0],
                        nnz_rows: group_size // bucket_size,
                        nnz_cols: bucket_size,
                    }
                ),
                ["A"],
                ["R", "I", "J"],
                ["R", "IO", "II", "J"],
                {"R": ["R"], "I": ["IO", "II"], "J": ["J"]},
                csf_to_ell3d_idx_map,
                csf_to_ell3d_inv_idx_map,
            )
        )
    mod = tvm.IRModule.from_expr(
        rgcn_hetero_forward.specialize(
            {
                rgcn_hetero_forward.params[-6]: m,
                rgcn_hetero_forward.params[-5]: n,
                rgcn_hetero_forward.params[-4]: num_rels,
                rgcn_hetero_forward.params[-3]: feat_size,
            }
        ).with_attr("horizontal_fuse", True)
    )
    mod = format_decompose(mod, rewrites)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    sch = tir.Schedule(mod["main"])
    for bucket_id, _ in enumerate(buckets):
        sp_iteration = sch.get_sparse_iteration("rgcn-hetero-forward_{}".format(bucket_id))
        fo, r, io, ii, j, fi = sch.get_sp_iters(sp_iteration)
        sch.sparse_reorder(sp_iteration, [r, io, ii, j, fo, fi])
        sch.sparse_fuse(sp_iteration, [r, io])
    mod = lower_sparse_iter(sch.mod)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    sch = tir.Schedule(mod["main"])
    for bucket_id, bucket_size in enumerate(buckets):
        blk = sch.get_block("rgcn-hetero-forward_{}0".format(bucket_id))
        io, ii, j, fo, fi = sch.get_loops(blk)
        foo, foi = sch.split(fo, [split_factor_f, None])
        sch.reorder(io, ii, foo, foi, j, fi)
        blk_outer, blk_inner = sch.blockize(j, True), blk
        read_W = sch.cache_read(blk_inner, 2, "local")
        write_Y = sch.cache_write(blk_inner, 0, "local")
        sch.annotate(write_Y, "atomic", True)
        sch.bind(fi, "threadIdx.x")
        sch.bind(sch.get_loops(read_W)[-1], "threadIdx.x")
        sch.unroll(j)
        sch.bind(foi, "threadIdx.y")
        sch.bind(io, "blockIdx.x")
        sch.unroll(ii)
        sch.unroll(foo)

    mod = lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    f = tvm.build(mod["main"], target="cuda")

    # prepare inputs
    dev = tvm.cuda(0)
    W_nd = tvm.nd.array(weight.cpu().view(-1), device=dev)
    X_nd = tvm.nd.array(feat.cpu().view(-1), device=dev)
    Y_nd = tvm.nd.array(th.zeros(m * feat_size), device=dev)
    args = [W_nd, X_nd, Y_nd]
    for bucket_id, _ in enumerate(buckets):
        args.append(
            tvm.nd.array(mask[bucket_id].numpy().reshape(-1).astype(np.float32), device=dev)
        )
        args.append(tvm.nd.array(row_indices[bucket_id].numpy().reshape(-1), device=dev))
        args.append(tvm.nd.array(col_indices[bucket_id].numpy().reshape(-1), device=dev))
    for bucket_id, _ in enumerate(buckets):
        args.append(tvm.nd.array(mids[bucket_id], device=dev))
    f(*args)

    tvm.testing.assert_allclose(Y_nd.numpy(), ground_truth_y.cpu().numpy().flatten(), rtol=1e-3)

    # evaluate time
    evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=10)
    print("sparse-tir:\t\t {:.3f} ms".format(evaluator(*args).mean * 1000))


if __name__ == "__main__":
    for feat_size in [32]:  # [4, 8, 16, 32]:
        for name in ["aifb", "mutag", "bgs", "am", "biokg"]:
            print("dataset {}, feat_size={}:".format(name, feat_size))
            dataset = get_hetero_dataset(name)
            g = dataset[0]
            type_pointers = get_type_pointers(g)
            n = g.num_nodes()
            r = len(g.etypes)
            feat = th.rand(n, feat_size).to(0) / 100
            weight = th.rand(r, feat_size, feat_size).to(0)
            # homograph
            ground_truth_y = get_ground_truth(g, type_pointers, feat, weight)
            test_rgcn_composable_format(
                g, type_pointers, feat_size, feat, weight, ground_truth_y, 16, 32, [1, 2, 4, 8, 16]
            )
