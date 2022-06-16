import dgl
import tvm
import argparse
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import torch as th
from tvm.script import tir as T
import tvm.sparse
from ogb.nodeproppred import DglNodePropPredDataset

@T.prim_func
def sddmm_new(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    feat_size: T.int32,
    chunk_size: T.int32,
    nnz_chunks: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m)
    JO = T.dense_variable(I, (n // chunk_size, nnz_chunks), indptr, "int32")
    JI = T.sparse_fixed(JO, (n, chunk_size), indices, "int32")
    K = T.dense_fixed(feat_size)
    J_detach = T.dense_fixed(n)
    A = T.match_sparse_buffer(a, (I, K), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K), "float32")
    C = T.match_sparse_buffer(c, (I, JO, JI), "float32")

    with T.iter([T.fuse(I, JO), JI, K], "SSSR", "sddmm") as [i, jo, ji, k]:
        with T.init():
            C[i, jo, ji] = 0.
        C[i, jo, ji] = C[i, jo, ji] + A[i, k] * B[ji, k]

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
    chunk_size = 4
    indptr, indices, _ = g.adj_sparse("csr")
    m = g.num_src_nodes()
    n = g.num_dst_nodes()
    nnz = g.number_of_edges()
    indptr = indptr.numpy()
    indices = indices.numpy()
    nnz_per_row = ((indptr[1:] - indptr[:-1]) + chunk_size - 1) // chunk_size
    indptr_new = np.concatenate([[0], nnz_per_row])
    indptr_new = np.cumsum(indptr_new)
    nnz_chunks = int(indptr_new[-1])

    M, N, F, CHUNK_SIZE, NNZ_CHUNKS = sddmm_new.params[-5:]
    a = th.rand(nnz_chunks, chunk_size, feat_size).to(th.float32)
    b = th.rand(n, feat_size).to(th.float32)
    c = th.zeros(nnz).to(th.float32)

    # tvm
    mod = tvm.IRModule.from_expr(sddmm_new.specialize({M: m, N: n, F: feat_size, CHUNK_SIZE: chunk_size, NNZ_CHUNKS: nnz_chunks}))
    mod = tvm.sparse.lower_sparse_iter(mod)

    # split preprocess and compute
    mod_preprocess = tvm.tir.transform.ExtractPreprocess()(mod)
    mod_sddmm = tvm.tir.transform.RemovePreprocess()(mod)

    ty = 4

    # schedule compute
    sch = tir.Schedule(mod_sddmm)
    blk = sch.get_block("sddmm0")
    jo, ji, k = sch.get_loops(blk)
    ko, kio, kii = sch.split(k, [None, 8, 4])
    rf_blk = sch.rfactor(kio, 2)
    jo = sch.get_loops(rf_blk)[0]
    joo, joi = sch.split(jo, [None, ty])
    sch.bind(joo, "blockIdx.x")
    sch.bind(joi, "threadIdx.y")
    sch.reverse_compute_at(blk, joi)
    sch.set_scope(rf_blk, 0, "local")
    read_A = sch.cache_read(rf_blk, 0, "local")
    read_B = sch.cache_read(rf_blk, 2, "local")
    write_C = sch.cache_write(blk, 0, "local")
    ji, ko, kio, kii = sch.get_loops(rf_blk)[-4:]
    sch.reorder(ko, ji, kio, kii)
    # TODO(zihao): fix here
    # schedule read A
    sch.compute_at(read_A, ko, True)
    print(sch.mod["main"].script())
    assert False
    ax0, ax1 = sch.split(sch.get_loops(read_A)[-1], [8, 4])
    sch.bind(ax0, "threadIdx.x")
    sch.vectorize(ax1)
    # schedule read B
    sch.compute_at(read_B, ji, True)
    ax0, ax1 = sch.split(sch.get_loops(read_B)[-1], [8, 4])
    sch.bind(ax0, "threadIdx.x")
    sch.vectorize(ax1)
    # schedule write C
    sch.reverse_compute_at(write_C, joi, True)
    ax0, ax1 = sch.get_loops(write_C)[-2:]
    sch.vectorize(ax1)
    # schedule rf
    sch.bind(kio, "threadIdx.x")
    sch.unroll(kii)
    sch.unroll(ko)
    # schedule write back
    ax0, ax1 = sch.get_loops(blk)[-2:]
    sch.reorder(ax1, ax0)
    sch.bind(ax0, "threadIdx.x")
    sch.unroll(ax1)
    # print(sch.mod["main"].script())
    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    sddmm = tvm.build(mod["main"], target="cuda")
    print(sddmm.imported_modules[0].get_source())
    assert False

    # compute mid
    a_nd = tvm.nd.array(a.view(-1).numpy(), tvm.cuda())
    b_nd = tvm.nd.array(b.view(-1).numpy(), tvm.cuda())
    c_nd = tvm.nd.array(c.numpy(), tvm.cuda())
    indptr_nd = tvm.nd.array(indptr.numpy(), tvm.cuda())
    indices_nd = tvm.nd.array(indices.numpy(), tvm.cuda())
    mid_nd = tvm.nd.array(np.zeros((nnz,), np.int32), tvm.cuda())

    preproc(a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd)

    # compute
    accum_time = 0.0
    runs = 0
    cold_start_time = 3
    for i in range(10):
        with TorchOpTimer() as timer:
            sddmm(a_nd, b_nd, c_nd, indptr_nd, indices_nd, mid_nd)
        # if i == 0:
        #     tvm.testing.assert_allclose(c_nd.numpy(), c_golden.view(-1).cpu(), rtol=1e-5)
        if i >= cold_start_time:
            accum_time += timer.time
            runs += 1

    print("Sparse-TIR:\t", accum_time / runs * 1000)


def get_dataset(name: str):
    if name == "arxiv":
        arxiv = DglNodePropPredDataset(name="ogbn-arxiv")
        g = arxiv[0][0]
    elif name == "proteins":
        proteins = DglNodePropPredDataset(name="ogbn-proteins")
        g = proteins[0][0]
    elif name == "pubmed":
        pubmed = dgl.data.PubmedGraphDataset()
        g = pubmed[0]
    elif name == "ppi":
        ppi = dgl.data.PPIDataset()
        g = dgl.batch(ppi)
    elif name == "reddit":
        reddit = dgl.data.RedditDataset()
        g = reddit[0]
    else:
        raise KeyError("Unknown dataset {}.".format(name))
    # g = dgl.graph(g.edges("uv", "srcdst"), num_nodes=g.num_nodes())
    return g.int()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("sddmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, default='pubmed', help="dataset name")
    args = parser.parse_args()
    name = args.dataset
    g = get_dataset(name)
    for feat_size in [64]:#[32, 64, 128, 256, 512]:
        bench_sddmm(g, feat_size)
