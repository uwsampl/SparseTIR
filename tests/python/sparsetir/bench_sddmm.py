import dgl
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import torch as th
from tvm.script import tir as T
import tvm.sparse
from ogb.nodeproppred import DglNodePropPredDataset
from sparse_tir_lowered_iter_scripts import fused_sddmm


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


def bench_sddmm(g: dgl.DGLGraph, feat_size: int):
    indptr, indices, _ = g.adj_sparse("csr")
    m = g.num_src_nodes()
    n = g.num_dst_nodes()
    nnz = g.number_of_edges()
    src_ids, _, eids = g.edges("all", "srcdst")
    
    M, N, F, NNZ = fused_sddmm.params[-4:]
    a = th.rand(m, feat_size).to(th.float32)
    b = th.rand(n, feat_size).to(th.float32)
    c = th.zeros(nnz).to(th.float32)

    # dgl
    accum_time = 0.0
    runs = 0
    cold_start_time = 3
    a_gpu = a.to(0)
    b_gpu = b.to(0)
    g = g.to(0)
    for i in range(10):
        with TorchOpTimer() as timer:
            c_golden = dgl.ops.u_dot_v(g, a_gpu, b_gpu)
        if i >= cold_start_time:
            accum_time += timer.time
            runs += 1
    print("dgl:\t\t", accum_time / runs * 1000)

    # tvm
    mod = tvm.IRModule.from_expr(fused_sddmm.specialize({M: m, N: n, F: feat_size, NNZ: nnz}))

    # split preprocess and compute
    mod_preprocess = tvm.tir.transform.ExtractPreprocess()(mod)
    mod_sddmm = tvm.tir.transform.RemovePreprocess()(mod)

    # schedule preprocess
    sch = tir.Schedule(mod_preprocess)
    blk = sch.get_block("binary_search_block_0_0")
    (i,) = sch.get_loops(blk)
    io, ii = sch.split(i, [None, 32])
    sch.bind(ii, "threadIdx.x")
    sch.bind(io, "blockIdx.x")
    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    preproc = tvm.build(mod["main"], target="cuda")

    # schedule compute
    sch = tir.Schedule(mod_sddmm)
    blk = sch.get_block("sddmm0")
    j, k = sch.get_loops(blk)
    ko, ki = sch.split(k, [None, 32])
    sch.bind(ki, "threadIdx.x")
    sch.unroll(ko)
    sch.bind(j, "blockIdx.x")
    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    sddmm = tvm.build(mod["main"], target="cuda")

    # compute mid
    a_nd = tvm.nd.array(a.view(-1).numpy(), tvm.gpu())
    b_nd = tvm.nd.array(b.view(-1).numpy(), tvm.gpu())
    c_nd = tvm.nd.array(c.numpy(), tvm.gpu())
    indptr_nd = tvm.nd.array(indptr.numpy(), tvm.gpu())
    indices_nd = tvm.nd.array(indices.numpy(), tvm.gpu())
    mid_nd = tvm.nd.array(np.zeros((nnz,), np.int32), tvm.gpu())

    preproc(a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd)
    assert np.allclose(src_ids.numpy(), mid_nd.numpy())

    # compute
    accum_time = 0.0
    runs = 0
    cold_start_time = 3
    for i in range(10):
        with TorchOpTimer() as timer:
            sddmm(a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd)
        if i == 0:
            tvm.testing.assert_allclose(c_nd.numpy(), c_golden.view(-1)[eids.long()].cpu())
        if i >= cold_start_time:
            accum_time += timer.time
            runs += 1

    print("Sparse-TIR:\t", accum_time / runs * 1000)


if __name__ == "__main__":
    arxiv = DglNodePropPredDataset(name="ogbn-arxiv")
    g = arxiv[0][0]
    for feat_size in [32, 64, 128, 256, 512]:
        bench_sddmm(g.int(), feat_size)
