import dgl
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import torch as th
from tvm.script import tir as T
from tvm.sparse import FormatRewriteRule, lower_sparse_buffer, lower_sparse_iter
import tvm.sparse
from ogb.nodeproppred import DglNodePropPredDataset
from sparse_tir_scripts import csrmm
from sparse_tir_format_rewrite_scripts import ell


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


def pad_graph(g: dgl.DGLGraph, tile_size=32) -> dgl.DGLGraph:
    u, v = g.edges()
    rows = [u.flatten()]
    cols = [v.flatten()]

    for node_id, deg in enumerate(g.in_degrees().tolist()):
        edges_to_add = ((deg + tile_size - 1) // tile_size) * tile_size - deg
        rows.append(th.full((edges_to_add,), 0))
        cols.append(th.full((edges_to_add,), node_id))

    rows = th.cat(rows)
    cols = th.cat(cols)

    return dgl.graph((rows, cols), num_nodes=g.num_dst_nodes())


@T.prim_func
def csrmm_tir(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    M: T.int32,
    N: T.int32,
    K: T.int32,
    NNZ: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (NNZ,), "float32")
    B = T.match_buffer(b, (N * K,), "float32")
    C = T.match_buffer(c, (M * K,), "float32")
    A_indptr = T.match_buffer(indptr, (M + 1,), "int32")
    A_indices = T.match_buffer(indices, (NNZ,), "int32")
    for i, k in T.grid(M, K):
        with T.block("spmm_outer"):
            vi, vk = T.axis.remap("SS", [i, k])
            with T.init():
                C[vi * K + vk] = 0.0
            for j in T.serial(0, A_indptr[vi + 1] - A_indptr[vi]):
                with T.block("spmm_inner"):
                    T.block_attr({"sparse": True})
                    vj = T.axis.R(NNZ, j + A_indptr[vi])
                    C[vi * K + vk] = C[vi * K + vk] + A_data[vj] * B[A_indices[vj] * K + vk]


@T.prim_func
def csrmm_padding_tir(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    M: T.int32,
    N: T.int32,
    K: T.int32,
    NNZT: T.int32,
    tile_size: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (NNZT * tile_size,), "float32")
    B = T.match_buffer(b, (N * K,), "float32")
    C = T.match_buffer(c, (M * K,), "float32")
    A_indptr = T.match_buffer(indptr, (M + 1,), "int32")
    A_indices = T.match_buffer(indices, (NNZT * tile_size,), "int32")
    for i, k in T.grid(M, K):
        with T.block("spmm_outer"):
            vi, vk = T.axis.remap("SS", [i, k])
            with T.init():
                C[vi * K + vk] = 0.0
            for j in T.grid(A_indptr[vi + 1] - A_indptr[vi]):
                with T.block("spmm_inner"):
                    T.block_attr({"sparse": True})
                    vj = T.axis.remap("R", [j])
                    for t in T.grid(tile_size):
                        with T.block("spmm_inner_2"):
                            vt = T.axis.remap("R", [t])
                            C[vi * K + vk] = (
                                C[vi * K + vk]
                                + A_data[(vj + A_indptr[vi]) * tile_size + vt]
                                * B[A_indices[(vj + A_indptr[vi]) * tile_size + vt] * K + vk]
                            )

def csr2ell_inv_index_map(o, i, j):
    return i, j

def csr2ell_index_map(i, j):
    return 0, i, j

def bench_hyb(g, feat_size=128, bucket_sizes=[]):
    # still work in progress
    in_degrees = g.in_degrees()
    # bucket_sizes = [4, 16, 64, 256]

    # rewrite csrmm
    nnz_cols_symbol = ell.params[-1]
    rewrites = []
    for bucket_size in bucket_sizes:
        rewrites.append(
            FormatRewriteRule(
                str(bucket_size),
                ell.specialize({nnz_cols_symbol: bucket_size}),
                ["A"],
                ["I", "J"],
                ["O", "I", "J"],
                {"I": ["O", "I"], "J": ["J"]},
                csr2ell_index_map,
                csr2ell_inv_index_map,
            )
        )
    mod = tvm.IRModule.from_expr(csrmm)
    mod = tvm.tir.transform.SparseFormatRewrite(rewrites)(mod)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    ell_rows = {}
    ell_n = {}
    ell_rows[bucket_sizes[0]] = ((in_degrees <= bucket_sizes[0])).nonzero().view(-1)
    for i in range(1, len(bucket_sizes) - 1):
        bucket_size = bucket_sizes[i]
        ell_rows[bucket_size] = ((in_degrees <= bucket_size) & (in_degrees > bucket_sizes[i - 1])).nonzero().view(-1)
    ell_rows[bucket_sizes[-1]] = (in_degrees > bucket_sizes[-2]).nonzero().view(-1)
    for bucket_size in bucket_sizes:
        ell_n[bucket_size] = len(ell_rows[bucket_size])
    n = g.num_nodes()
    nnz = g.num_edges()

    ell_indices = {}
    ell_a = {}
    for bucket_size in bucket_sizes[:-1]:
        indices = []
        a = []
        for row in ell_rows[bucket_size]:
            in_edges = g.in_edges([row])[0]
            indices.append(th.cat([in_edges, th.full((bucket_size - len(in_edges),), 0)]))
            a.append(th.cat([th.ones(len(in_edges)), th.zeros(bucket_size - len(in_edges))]))
        ell_indices[bucket_size] = th.stack(indices)
        ell_a[bucket_size] = th.stack(a)

    # split rows for the last bucket
    indices = []
    a = []
    new_rows = []
    bucket_size = bucket_sizes[-1]
    for row in ell_rows[bucket_size]:
        in_edges = g.in_edges([row])[0]
        for i in range((len(in_edges) + bucket_size - 1) // bucket_size):
            in_edges_i = in_edges[i * bucket_size : (i + 1) * bucket_size]
            indices.append(th.cat([in_edges_i, th.full((bucket_size - len(in_edges_i),), 0)]))
            a.append(th.cat([th.ones(len(in_edges_i)), th.zeros(bucket_size - len(in_edges_i))]))
            new_rows.append(row)
    ell_indices[bucket_size] = th.stack(indices)
    ell_a[bucket_size] = th.stack(a)
    ell_rows[bucket_size] = th.tensor(new_rows).int()
    ell_n[bucket_size] = len(new_rows)

    params = mod["main"].params
    param_map = {
        params[5]: g.num_dst_nodes(), # m
        params[6]: g.num_src_nodes(), # n
        params[7]: feat_size, # feat_size,
        params[8]: nnz, # nnz
    }
    for i in range(len(bucket_sizes)):
        bucket_size = bucket_sizes[i]
        param_map[params[9 + 7 * i + 4]] = g.num_dst_nodes()
        param_map[params[9 + 7 * i + 5]] = g.num_src_nodes()
        param_map[params[9 + 7 * i + 6]] = ell_n[bucket_size]

    mod["main"] = mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)
    sch = tvm.tir.Schedule(mod)
    for sp_iter_name in ['csrmm_{}'.format(bucket_size) for bucket_size in bucket_sizes]:
        sp_iteration = sch.get_sparse_iteration(sp_iter_name)
        o, i, j, k = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [o, i])
    mod = sch.mod
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    for bucket_size in bucket_sizes[:-1]:
        blk = sch.get_block("csrmm_{}0".format(bucket_size))
        i, j, f = sch.get_loops(blk)
        foo, foi, fi = sch.split(f, [None, 2, 32])
        sch.bind(fi, "threadIdx.x")
        sch.unroll(foi)
        sch.bind(foo, "blockIdx.y")
        jo, ji = sch.split(j, [None, 32])
        sch.unroll(ji)
        io, ii = sch.split(i, [None, max(1, 32 // bucket_size)])
        sch.bind(io, "blockIdx.x")

    # schedule last bucket
    blk = sch.get_block("csrmm_{}0".format(bucket_sizes[-1]))
    i, j, f = sch.get_loops(blk)
    sch.annotate(blk, "atomic", True)
    write_blk = sch.cache_write(blk, 0, "local")
    sch.reverse_compute_at(write_blk, i)
    foo, foi, fi = sch.split(f, [None, 2, 32])
    sch.bind(fi, "threadIdx.x")
    sch.unroll(foi)
    sch.bind(foo, "blockIdx.y")
    jo, ji = sch.split(j, [None, 32])
    sch.unroll(ji)
    io, ii = sch.split(i, [None, max(1, 32 // bucket_size)])
    sch.bind(io, "blockIdx.x")
    ax0, ax1, ax2 = sch.split(sch.get_loops(write_blk)[-1], [None, 2, 32])
    sch.bind(ax0, "blockIdx.y")
    sch.bind(ax2, "threadIdx.x")
    sch.unroll(ax1)

    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    f = tvm.build(mod, target="cuda")

    b_nd = tvm.nd.array(np.ones(n * feat_size,).astype("float32"), device=tvm.cuda(0))
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float32"), device=tvm.cuda(0))
    ell_indices_i_nd = {}
    ell_a_nd = {}
    ell_indices_j_nd = {}
    for bucket_size in bucket_sizes:
        ell_indices_i_nd[bucket_size] = tvm.nd.array(ell_rows[bucket_size].numpy().astype("int32"), device=tvm.cuda(0))
        ell_a_nd[bucket_size] = tvm.nd.array(ell_a[bucket_size].view(-1).numpy().astype("float32"), device=tvm.cuda(0))
        ell_indices_j_nd[bucket_size] = tvm.nd.array(ell_indices[bucket_size].view(-1).numpy().astype("int32"), device=tvm.cuda(0))

    accum_time = 0.0
    runs = 0
    cold_start_time = 3
    args = [b_nd, c_nd]
    for bucket_size in bucket_sizes:
        args += [ell_a_nd[bucket_size], ell_indices_i_nd[bucket_size], ell_indices_j_nd[bucket_size]]
    print(ell_n)
    for i in range(10):
        with TorchOpTimer() as timer:
            f(*args)
        if i >= cold_start_time:
            accum_time += timer.time
            runs += 1

    print(accum_time / runs * 1000)


def bench_tir_csrmm(g, feat_size=128):
    # generate random input
    indptr, indices, _ = g.adj_sparse("csc")

    m = g.num_src_nodes()
    n = g.num_dst_nodes()
    k = feat_size
    nnz = g.num_edges()
    x = np.random.rand(n, k).astype("float32")
    y = np.zeros((m * k,)).astype("float32")

    # specialize function
    _, _, _, _, _, M, N, K, NNZ = csrmm_tir.params
    sch = tir.Schedule(csrmm_tir.specialize({M: m, N: n, K: k, NNZ: nnz}))
    blk_outer = sch.get_block("spmm_outer")
    i, k = sch.get_loops(blk_outer)
    sch.bind(i, "blockIdx.x")
    sch.bind(k, "threadIdx.x")

    # convert numpy tensor to tvm ndarray
    A_indptr = tvm.nd.array(indptr.numpy().astype("int32"), device=tvm.cuda(0))
    A_indices = tvm.nd.array(indices.numpy().astype("int32"), device=tvm.cuda(0))
    A_data = tvm.nd.array(np.ones((nnz,)).astype("float32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod, target="cuda")
    accum_time = 0.0
    runs = 0
    cold_start_time = 3
    for i in range(10):
        with TorchOpTimer() as timer:
            f(A_data, X_nd, Y_nd, A_indptr, A_indices)
        if i >= cold_start_time:
            accum_time += timer.time
            runs += 1
        Y_val = Y_nd.numpy()
    print("tir naive time: {:.3f}ms".format(accum_time / runs * 1000))

    g_gpu = g.to(0)
    h = th.from_numpy(x).to(0)
    weight = th.ones(nnz).to(0)
    accum_time = 0.0
    runs = 0
    cold_start_time = 3
    for i in range(10):
        with TorchOpTimer() as timer:
            out = dgl.ops.u_mul_e_sum(g_gpu, h, weight)
        if i >= cold_start_time:
            accum_time += timer.time
            runs += 1
    print("cusparse time: {:.3f}ms".format(accum_time / runs * 1000))

    for tile_size in [8, 16, 24, 32, 40, 48, 56, 64]:
        g_pad = pad_graph(g, tile_size=tile_size)
        m = g_pad.num_src_nodes()
        n = g_pad.num_dst_nodes()
        k = feat_size
        x = np.random.rand(n, k).astype("float32")
        y = np.zeros((m * k,)).astype("float32")
        nnzt = g_pad.num_edges() // tile_size
        indptr_, indices_, _ = g_pad.adj_sparse("csc")
        _, _, _, _, _, M, N, K, NNZT, TILE_SIZE = csrmm_padding_tir.params
        sch = tir.Schedule(
            csrmm_padding_tir.specialize({M: m, N: n, K: k, NNZT: nnzt, TILE_SIZE: tile_size})
        )
        # print(sch.mod["main"].script())
        blk_outer = sch.get_block("spmm_outer")
        i, k = sch.get_loops(blk_outer)
        koo, ko, ki = sch.split(k, [None, 2, 32])
        blk_inner_2 = sch.get_block("spmm_inner_2")
        (t,) = sch.get_loops(blk_inner_2)
        sch.unroll(t)
        sch.bind(ko, "vthread.x")
        sch.bind(i, "blockIdx.x")
        sch.bind(koo, "blockIdx.y")
        sch.bind(ki, "threadIdx.x")

        # convert numpy tensor to tvm ndarray
        A_indptr = tvm.nd.array((indptr_.numpy() // tile_size).astype("int32"), device=tvm.cuda(0))
        A_indices = tvm.nd.array(indices_.numpy().astype("int32"), device=tvm.cuda(0))
        A_data = tvm.nd.array(np.ones((nnzt * tile_size,)).astype("float32"), device=tvm.cuda(0))
        X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
        Y_nd = tvm.nd.array(y, device=tvm.cuda(0))

        # build function
        f = tvm.build(sch.mod, target="cuda")
        # print(f.imported_modules[0].get_source())
        accum_time = 0.0
        runs = 0
        cold_start_time = 3
        for i in range(10):
            with TorchOpTimer() as timer:
                f(A_data, X_nd, Y_nd, A_indptr, A_indices)
            if i >= cold_start_time:
                accum_time += timer.time
                runs += 1
            Y_val = Y_nd.numpy()
        print(
            "tir w/ padding (tile_size={}) time: {:.3f}ms".format(
                tile_size, accum_time / runs * 1000
            )
        )


if __name__ == "__main__":
    arxiv = DglNodePropPredDataset(name="ogbn-arxiv")
    g = arxiv[0][0] # [1, 2, 4, 8, 16, 32, 64, 128]
    # proteins = DglNodePropPredDataset(name='ogbn-proteins')
    # g = proteins[0][0]
    # pubmed = dgl.data.PubmedGraphDataset()
    # g = pubmed[0] # [1, 8, 16]
    # ppi = dgl.data.PPIDataset()
    # g = dgl.batch(ppi)
    # reddit = dgl.data.RedditDataset()
    # g = reddit[0] # [64, 128, 256, 512]

    for feat_size in [32, 64, 128, 256, 512]:
        print("feat_size=", feat_size)
        bench_hyb(g, feat_size=feat_size, bucket_sizes=[1, 2, 4, 8, 16, 32, 64, 128])
    for feat_size in [32, 64, 128, 256, 512]:
        print("feat_size=", feat_size)
        bench_tir_csrmm(g, feat_size=feat_size)
