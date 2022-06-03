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


def csr2ell_inv_index_map(o, i, j):
    return i, j


def csr2ell_index_map(i, j):
    return 0, i, j


cached_bucketing_format = []


def bench_hyb(
    g,
    x,
    y_golden,
    feat_size=128,
    bucket_sizes=[],
    cwm=2,
    column_part=1,
):
    global cached_bucketing_format
    mat = g.adj(transpose=True, scipy_fmt="csr")
    del g
    cwm = min(cwm, feat_size // 32)
    buckets = bucket_sizes * column_part
    m = mat.shape[0]
    n = mat.shape[1]
    nnz = mat.nnz
    per_column_part_size = (n + column_part - 1) // column_part

    num_buckets = len(buckets)
    ell_n = []
    is_bucket_atomic = []

    for partition in range(column_part):
        sub_mat = mat[:, partition * per_column_part_size : (partition + 1) * per_column_part_size]
        in_degrees = sub_mat.indptr[1:] - sub_mat.indptr[:-1]
        for i, bucket_size in enumerate(bucket_sizes[:-1]):
            last_bucket_size = 0 if i == 0 else bucket_sizes[i - 1]
            ell_n.append(int(((in_degrees > last_bucket_size) & (in_degrees <= bucket_size)).sum()))
            if column_part == 1:
                is_bucket_atomic.append(False)
            else:
                is_bucket_atomic.append(True)
        sub_indegrees = in_degrees[in_degrees > bucket_sizes[-2]]
        ell_n.append(int(((sub_indegrees + bucket_sizes[-1] - 1) // bucket_sizes[-1]).sum()))
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
            sch.unroll(sch.get_loops(write_blk)[-2])
        sch.bind(fi, "threadIdx.x")
        sch.bind(foo, "blockIdx.y")
        sch.unroll(foi)
        sch.unroll(j)
        io, ii = sch.split(i, [None, max(1, bucket_sizes[-1] // bucket_size)])
        sch.bind(io, "blockIdx.x")
        init_blk = sch.decompose_reduction(blk, fi)
        ax0, ax1 = sch.get_loops(init_blk)[-2:]
        sch.bind(ax0, "threadIdx.x")
        sch.unroll(ax1)

    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    f = tvm.build(mod, target="cuda")

    # prepare new formats
    if len(cached_bucketing_format) > 0:
        ell_indices, ell_a, ell_rows = cached_bucketing_format
    else:
        ell_rows = []
        ell_indices = []
        ell_a = []

        for partition in range(column_part):
            sub_mat = mat[
                :, partition * per_column_part_size : (partition + 1) * per_column_part_size
            ]
            in_degrees = sub_mat.indptr[1:] - sub_mat.indptr[:-1]

            for i, bucket_size in enumerate(bucket_sizes[:-1]):
                last_bucket_size = 0 if i == 0 else bucket_sizes[i - 1]
                ell_rows.append(
                    ((in_degrees > last_bucket_size) & (in_degrees <= bucket_size)).nonzero()[0]
                )
            ell_rows.append((in_degrees > bucket_sizes[-2]).nonzero()[0])

            for i, bucket_size in enumerate(bucket_sizes[:-1]):
                indices = np.zeros(
                    (ell_n[partition * len(bucket_sizes) + i], bucket_size), dtype=np.int32
                )
                a = np.zeros(
                    (ell_n[partition * len(bucket_sizes) + i], bucket_size), dtype=np.float32
                )
                for j, row_id in enumerate(ell_rows[partition * len(bucket_sizes) + i]):
                    row = sub_mat[row_id]
                    indices[j, : row.nnz] = row.indices + partition * per_column_part_size
                    a[j, : row.nnz] = row.data
                ell_indices.append(indices)
                ell_a.append(a)

            # split rows for the last bucket
            indices = np.zeros(
                (ell_n[(partition + 1) * len(bucket_sizes) - 1], bucket_sizes[-1]), dtype=np.int32
            )
            a = np.zeros(
                (ell_n[(partition + 1) * len(bucket_sizes) - 1], bucket_sizes[-1]), dtype=np.float32
            )
            new_rows = np.zeros((ell_n[(partition + 1) * len(bucket_sizes) - 1],), dtype=np.int32)
            bucket_size = bucket_sizes[-1]
            i = 0
            for row_id in ell_rows[-1]:
                row = sub_mat[row_id]
                for start_offset in range(0, row.nnz, bucket_size):
                    if start_offset + bucket_size >= row.nnz:
                        # last bucket
                        indices[i, : row.nnz - start_offset] = (
                            row.indices[start_offset:] + partition * per_column_part_size
                        )
                        a[i, : row.nnz - start_offset] = row.data[start_offset:]
                    else:
                        indices[i] = (
                            row.indices[start_offset : start_offset + bucket_size]
                            + partition * per_column_part_size
                        )
                        a[i].fill(1)
                    new_rows[i] = row_id
                    i += 1

            ell_indices.append(indices)
            ell_a.append(a)
            ell_rows[-1] = new_rows

        cached_bucketing_format = ell_indices, ell_a, ell_rows

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
        ell_indices_i_nd.append(tvm.nd.array(ell_rows[i].astype("int32"), device=tvm.cuda(0)))
        ell_a_nd.append(tvm.nd.array(ell_a[i].reshape(-1).astype("float32"), device=tvm.cuda(0)))
        ell_indices_j_nd.append(
            tvm.nd.array(ell_indices[i].reshape(-1).astype("int32"), device=tvm.cuda(0))
        )

    # prepare args
    args = [b_nd, c_nd]
    for i in range(num_buckets):
        args += [ell_a_nd[i], ell_indices_i_nd[i], ell_indices_j_nd[i]]

    # test accuracy
    f(*args)
    tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)

    # evaluate time
    evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=10)
    print("tir hyb time: {:.3f}ms".format(evaluator(*args).mean * 1000))


col_part_config = {"arxiv": 1, "proteins": 8, "pubmed": 1, "ppi": 8, "reddit": 8}

bucketing_config = {
    "arxiv": [1, 2, 4, 8, 16, 32],
    "proteins": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "pubmed": [1, 2, 4, 8, 16, 32],
    "ppi": [1, 2, 4, 8, 16, 32],
    "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
}


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
    return g.int()


if __name__ == "__main__":
    name = "arxiv"
    g = get_dataset(name)

    for feat_size in [32, 64, 128, 256, 512]:
        print("feat_size =", feat_size)
        x = th.rand((g.num_src_nodes(), feat_size))
        y_golden = dgl.ops.copy_u_sum(g, x)
        bench_hyb(
            g,
            x,
            y_golden,
            feat_size=feat_size,
            bucket_sizes=bucketing_config[name],
            cwm=2,
            column_part=col_part_config[name],
        )
