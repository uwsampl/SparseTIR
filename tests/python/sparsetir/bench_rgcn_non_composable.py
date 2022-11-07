from dgl.heterograph import DGLHeteroGraph
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
from sparse_tir_scripts import rgcn_hetero_forward
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer
from typing import List, Tuple
from utils import get_hetero_dataset


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


def test_lower_rgcn_hetero(
    g: dgl.DGLHeteroGraph,
    feat_size: int,
    feat,
    weight,
    split_factor_f: int,
):
    M, N, R, FEAT_SIZE, NNZ_I, NNZ_J = rgcn_hetero_forward.params[-6:]
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    r = len(g.etypes)
    nnz_j = g.num_edges()

    out = np.zeros((m * feat_size))
    A = tvm.nd.array(np.zeros((nnz_j,), dtype=np.float32), device=tvm.cuda(0))
    W = tvm.nd.array(weight.astype("float32"), device=tvm.cuda(0))
    X = tvm.nd.array(feat.astype("float32"), device=tvm.cuda(0))
    Y = tvm.nd.array(out.astype("float32"), device=tvm.cuda(0))

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
        indptr_i.append(th.LongTensor([len(unique_nodes) + indptr_i[-1].item()]))
        indices_i.append(unique_nodes + g.ntype_pointer[dst_type_id])
        indptr_j.append(indptr[unique_nodes + 1] + g.etype_pointer[etype_id])
        indices_j.append(indices + g.ntype_pointer[src_type_id])

    indptr_i = tvm.nd.array(th.cat(indptr_i).numpy().astype("int32"), device=tvm.cuda(0))
    indices_i = tvm.nd.array(th.cat(indices_i).numpy().astype("int32"), device=tvm.cuda(0))
    indptr_j = tvm.nd.array(th.cat(indptr_j).numpy().astype("int32"), device=tvm.cuda(0))
    indices_j = tvm.nd.array(th.cat(indices_j).numpy().astype("int32"), device=tvm.cuda(0))

    nnz_i = indices_i.shape[0]
    mod = tvm.IRModule.from_expr(
        rgcn_hetero_forward.specialize(
            {M: m, N: n, R: r, FEAT_SIZE: feat_size, NNZ_I: nnz_i, NNZ_J: nnz_j}
        )
    )
    mod = lower_sparse_iter(mod)
    sch = tir.Schedule(mod)

    blk0 = sch.get_block("rgcn-hetero-forward0")
    blk1 = sch.get_block("rgcn-hetero-forward1")
    blk2 = sch.get_block("rgcn-hetero-forward2")
    read_blk = sch.cache_read(blk1, 3, "local")
    write_blk = sch.cache_write(blk2, 0, "local")
    sch.annotate(write_blk, "atomic", True)
    fo, r = sch.get_loops(blk0)
    foo, foi = sch.split(fo, [split_factor_f, None])
    (i,) = sch.get_loops(blk1)
    (io, ii) = sch.split(i, [256, None])
    j, f_in = sch.get_loops(blk2)
    sch.reorder(f_in, j)
    sch.bind(r, "blockIdx.y")
    sch.bind(io, "blockIdx.x")
    sch.bind(foo, "blockIdx.z")
    sch.bind(f_in, "threadIdx.x")
    sch.bind(foi, "threadIdx.y")
    _, _, ax2 = sch.get_loops(read_blk)
    sch.bind(ax2, "threadIdx.x")
    mod = lower_sparse_buffer(sch.mod)
    f = tvm.build(mod["main"], target="cuda")
    # print(f.imported_modules[0].get_source())

    args = [A, W, X, Y, indptr_i, indices_i, indptr_j, indices_j]
    f(*args)

    # evaluate time
    evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=10)
    print("sparse-tir: {:.3f}".format(evaluator(*args).mean * 1000))


if __name__ == "__main__":
    for feat_size in [32]:  # [4, 8, 16, 32, 64]:
        for name in ['aifb', 'mutag', 'bgs', 'am', 'biokg']:
            dataset = get_hetero_dataset(name)
            g = dataset[0]
            type_pointers = prepare_hetero_graph_simplified(g)
            n = g.num_nodes()
            r = len(g.etypes)
            feat = th.rand(n, feat_size).to(0) / 100
            weight = th.rand(r, feat_size, feat_size).to(0)
            # heterograph
            g.ntype_pointer = type_pointers["ntype_node_pointer"]
            g.etype_pointer = type_pointers["etype_edge_pointer"]
            for split_factor_f in [1, 2, 4, 8, 16, 32]:
                print(
                    "dataset {}, split_factor_f {}:".format(
                        name, split_factor_f,
                    )
                )
                test_lower_rgcn_hetero(
                    g,
                    feat_size,
                    feat.view(-1).cpu().numpy(),
                    weight.view(-1).cpu().numpy(),
                    split_factor_f,
                )
