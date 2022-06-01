import dgl
from sklearn.metrics import jaccard_score
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
from sparse_tir_format_rewrite_scripts import ell


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
    A = T.match_sparse_buffer(a, (I, J), "float32")
    B = T.match_sparse_buffer(b, (J_detach, K1, K2, K3), "float32")
    C = T.match_sparse_buffer(c, (I, K1, K2, K3), "float32")
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


cached_formats = []

def bench_hyb(g, x, y_golden, feat_size=128, bucket_sizes=[], cwm=2):
    global cached_formats
    mat = g.adj(transpose=True, scipy_fmt='csr')
    del g
    cwm = min(cwm, feat_size // 32)
    buckets = bucket_sizes
    num_buckets = len(buckets)
    m = mat.shape[0]
    n = mat.shape[1]
    nnz = mat.nnz

    in_degrees = mat.indptr[1:] - mat.indptr[:-1]
    ell_n = []
    is_bucket_atomic = []
    for bucket_size in bucket_sizes[:-1]:
        ell_n.append(int((in_degrees <= bucket_size).sum()))
        is_bucket_atomic.append(False)
    for i in range(1, len(bucket_sizes) - 1):
        ell_n[i] = ell_n[i] - sum(ell_n[:i])
    sub_indegrees = in_degrees[in_degrees > bucket_sizes[-2]]
    ell_n.append(
        int(((sub_indegrees + bucket_sizes[-1] - 1) // bucket_sizes[-1]).sum())
    )
    is_bucket_atomic.append(True)
    print(ell_n)

    # rewrite csrmm
    nnz_cols_symbol = ell.params[-1]
    rewrites = []
    for i, bucket_size in enumerate(buckets):
        rewrites.append(
            FormatRewriteRule(
                str(i),
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

    # specialize
    params = mod["main"].params
    param_map = {
        params[5]: m,  # m
        params[6]: n,  # n
        params[7]: feat_size // cwm // 32,  # num_tiles,
        params[8]: nnz,  # nnz
        params[9]: cwm,  # cwm
    }
    for i, bucket_size in enumerate(buckets):
        param_map[params[10 + 7 * i + 4]] = m 
        param_map[params[10 + 7 * i + 5]] = n 
        param_map[params[10 + 7 * i + 6]] = ell_n[i]

    mod["main"] = mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)

    # schedule
    sch = tvm.tir.Schedule(mod)
    for sp_iter_name in ["csrmm_{}".format(i) for i in range(num_buckets)]:
        sp_iteration = sch.get_sparse_iteration(sp_iter_name)
        o, i, j, k1, k2, k3 = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [o, i])

    mod = sch.mod
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    for i, bucket_size in enumerate(buckets):
        is_atomic = is_bucket_atomic[i]
        bucket_size = buckets[i]
        blk = sch.get_block("csrmm_{}0".format(i))
        i, j, foo, foi, fi = sch.get_loops(blk)
        sch.reorder(foo, fi, j, foi)
        if is_atomic:
            sch.annotate(blk, "atomic", True)
            write_blk = sch.cache_write(blk, 0, "local")
            sch.reverse_compute_at(write_blk, fi, True)
            ax = sch.get_loops(write_blk)[-1]
            sch.unroll(ax)
        sch.unroll(foi)
        sch.bind(fi, "threadIdx.x")
        sch.bind(foo, "blockIdx.y")
        sch.unroll(j)
        io, ii = sch.split(i, [None, max(1, 32 // bucket_size)])
        sch.bind(io, "blockIdx.x")

    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    f = tvm.build(mod, target="cuda")
    # print(f.imported_modules[0].get_source())

    # prepare new formats
    if len(cached_formats) > 0:
        ell_indices, ell_a, ell_rows = cached_formats
    else:
        ell_rows = []
        ell_rows.append(((in_degrees <= bucket_sizes[0])).nonzero()[0])
        for i in range(1, len(bucket_sizes) - 1):
            bucket_size = bucket_sizes[i]
            ell_rows.append(
                ((in_degrees <= bucket_size) & (in_degrees > bucket_sizes[i - 1]))
                .nonzero()[0]
            )
        ell_rows.append((in_degrees > bucket_sizes[-2]).nonzero()[0])

        ell_indices = []
        ell_a = []
        for i, bucket_size in enumerate(buckets[:-1]):
            indices = np.zeros((ell_n[i], bucket_size), dtype=np.int32)
            a = np.zeros((ell_n[i], bucket_size), dtype=np.float32)
            for i, row in enumerate(ell_rows[i]):
                mat_row = mat[row]
                indices[i, :mat_row.nnz] = mat_row.indices
                a[i, :mat_row.nnz] = mat_row.data
            ell_indices.append(indices)
            ell_a.append(a)

        # split rows for the last bucket
        indices = np.zeros((ell_n[-1], buckets[-1]), dtype=np.int32)
        a = np.zeros((ell_n[-1], buckets[-1]), dtype=np.float32)
        new_rows = np.zeros((ell_n[-1],), dtype=np.int32)
        bucket_size = bucket_sizes[-1]
        i = 0
        for row in ell_rows[-1]:
            mat_row = mat[row]
            for start_offset in range(0, mat_row.nnz, bucket_size):
                if start_offset + bucket_size >= mat_row.nnz:
                    # last bucket
                    indices[i, :mat_row.nnz - start_offset] = mat_row.indices[start_offset:]
                    a[i, :mat_row.nnz - start_offset] = mat_row.data[start_offset:]
                else:
                    indices[i] = mat_row.indices[start_offset: start_offset + bucket_size]
                    a[i].fill(1)
                new_rows[i] = row
                i += 1

        ell_indices.append(indices)
        ell_a.append(a)
        ell_rows[-1] = new_rows
        cached_formats = ell_indices, ell_a, ell_rows

    # prepare nd array
    b_nd = tvm.nd.array(
        x.numpy().reshape(-1).astype("float32"),
        device=tvm.cuda(0),
    )
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float32"), device=tvm.cuda(0))
    ell_indices_i_nd = []
    ell_a_nd = []
    ell_indices_j_nd = []
    for i in range(num_buckets):
        ell_indices_i_nd.append(
            tvm.nd.array(ell_rows[i].astype("int32"), device=tvm.cuda(0))
        )
        ell_a_nd.append(
            tvm.nd.array(ell_a[i].reshape(-1).astype("float32"), device=tvm.cuda(0))
        )
        ell_indices_j_nd.append(
            tvm.nd.array(ell_indices[i].reshape(-1).astype("int32"), device=tvm.cuda(0))
        )

    # prepare args
    args = [b_nd, c_nd]
    for i in range(num_buckets):
        args += [ell_a_nd[i], ell_indices_i_nd[i], ell_indices_j_nd[i]]
    
    # test accuracy
    f(*args)
    tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy())

    # evaluate time
    evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=10)
    print("tir hyb time: {:.3f}ms".format(evaluator(*args).mean * 1000))
    

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

    for tile_size in [4, 8, 16, 24, 32, 40, 48, 56, 64]:
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
    g = arxiv[0][0] # [1, 2, 4, 8, 16, 32]
    # proteins = DglNodePropPredDataset(name="ogbn-proteins")
    # g = proteins[0][0]
    # pubmed = dgl.data.PubmedGraphDataset()
    # g = pubmed[0] # [1, 8, 16]
    # ppi = dgl.data.PPIDataset()
    # g = dgl.batch(ppi) # [4, 8, 16, 32]
    # reddit = dgl.data.RedditDataset()
    # g = reddit[0] # [64, 128, 256, 512]

    g = g.int()
    for feat_size in [32, 64, 128, 256, 512]:
        print("feat_size=", feat_size)
        x = th.ones((g.num_src_nodes(), feat_size))
        y_golden = dgl.ops.copy_u_sum(g, x)
        bench_hyb(g, x, y_golden, feat_size=feat_size, bucket_sizes=[1, 2, 4, 8, 16, 32], cwm=2)
    for feat_size in [32, 64, 128, 256, 512]:
        print("feat_size=", feat_size)
        bench_tir_csrmm(g, feat_size=feat_size)
