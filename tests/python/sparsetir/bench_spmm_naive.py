import dgl
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import argparse
import numpy as np
import torch as th
from tvm.script import tir as T
from tvm.sparse import FormatRewriteRule, lower_sparse_buffer, lower_sparse_iter
import tvm.sparse
from ogb.nodeproppred import DglNodePropPredDataset
from sparse_tir_format_rewrite_scripts import ell, padding


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    num_tiles: T.int32,
    nnz: T.int32,
    cwm: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), "int32")
    J_detach = T.dense_fixed(n)
    K1 = T.dense_fixed(num_tiles)
    K2 = T.dense_fixed(cwm)
    K3 = T.dense_fixed(32)
    A = T.match_sparse_buffer(a, (I, J), "float16")
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float16")
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float16")
    with T.iter([I, J, K1, K2, K3], "SRSSS", "csrmm") as [i, j, k1, k2, k3]:
        with T.init():
            C[i, k1, k2, k3] = 0.0
        C[i, k1, k2, k3] = C[i, k1, k2, k3] + A[i, j] * B[j, k1, k2, k3]


class TorchOpTimer(object):
    def __enter__(self):
        self.start_event = th.cuda.Event(enable_timing=True)
        self.end_event = th.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        th.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event) / 1e3


def bench_hyb(
    g,
    x,
    y_golden,
    feat_size=128,
    cwm=2,
):
    indptr, indices, _ = g.adj_sparse('csc')
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    nnz = g.num_edges()
    if feat_size < 64:
        cwm = 1
    mod = tvm.IRModule.from_expr(csrmm)
    # specialize
    params = mod["main"].params
    param_map = {
        params[5]: m,  # m
        params[6]: n,  # n
        params[7]: feat_size // cwm // 32,  # num_tiles,
        params[8]: nnz,  # nnz
        params[9]: cwm,  # cwm
    }

    mod["main"] = mod["main"].specialize(param_map)

    # schedule
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    outer_blk = sch.get_block("csrmm0")
    inner_blk = sch.get_block("csrmm1")
    i, = sch.get_loops(outer_blk)
    j, foo, foi, fi = sch.get_loops(inner_blk)
    sch.reorder(foo, fi, j, foi)
    sch.bind(fi, "threadIdx.x")
    sch.bind(foo, "blockIdx.y")
    sch.unroll(foi)
    io, ii = sch.split(i, [None, 8])
    sch.bind(io, "blockIdx.x")
    sch.bind(ii, "threadIdx.y")
    init_blk = sch.decompose_reduction(inner_blk, fi)
    ax0, ax1 = sch.get_loops(init_blk)[-2:]
    sch.bind(ax0, "threadIdx.x")
    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    f = tvm.build(mod["main"], target='cuda')
    # prepare nd array
    indptr_nd = tvm.nd.array(indptr.numpy().astype("int32"), device=tvm.cuda(0))
    b_nd = tvm.nd.array(
        x.numpy().reshape(-1).astype("float16"),
        device=tvm.cuda(0),
    )
    indices_nd = tvm.nd.array(indices.numpy().astype("int32"), device=tvm.cuda(0))
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float16"), device=tvm.cuda(0))
    a_nd = tvm.nd.array(np.ones((nnz,)).astype("float16"), device=tvm.cuda(0))
    args = [a_nd, b_nd, c_nd, indptr_nd, indices_nd]
    f(*args)
    #tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)
    #evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=100)
    #print("tir naive time: {:.5f} ms".format(evaluator(*args).mean * 1000))


def get_dataset(name: str):
    if name == "arxiv":
        arxiv = DglNodePropPredDataset(name="ogbn-arxiv")
        g = arxiv[0][0]
    elif name == "proteins":
        proteins = DglNodePropPredDataset(name="ogbn-proteins")
        g = proteins[0][0]
    elif name == "products":
        products = DglNodePropPredDataset(name="ogbn-products")
        g = products[0][0]
    elif name == "pubmed":
        pubmed = dgl.data.PubmedGraphDataset()
        g = pubmed[0]
    elif name == "citeseer":
        citeseer = dgl.data.CiteseerGraphDataset()
        g = citeseer[0]
    elif name == "cora":
        cora = dgl.data.CoraGraphDataset()
        g = cora[0]
    elif name == "ppi":
        ppi = dgl.data.PPIDataset()
        g = dgl.batch(ppi)
    elif name == "reddit":
        reddit = dgl.data.RedditDataset()
        g = reddit[0]
    else:
        raise KeyError("Unknown dataset {}.".format(name))
    return g.int()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, default='arxiv', help="dataset name")
    args = parser.parse_args()
    name = args.dataset
    g = get_dataset(name)

    #for feat_size in [32, 64, 128, 256, 512]:
    for feat_size in [256]:
        print("feat_size =", feat_size)
        x = th.rand((g.num_src_nodes(), feat_size))
        y_golden = dgl.ops.copy_u_sum(g, x)
        bench_hyb(
            g,
            x,
            y_golden,
            feat_size=feat_size,
            cwm=2,
        )
