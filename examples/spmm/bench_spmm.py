import dgl
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import argparse
import numpy as np
import torch as th
from tvm.script import tir as T
from tvm.sparse import (
    FormatRewriteRule,
    lower_sparse_buffer,
    lower_sparse_iter,
    column_part_hyb,
    format_decompose,
)
import tvm.sparse
from utils import get_dataset, ell


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


cached_bucketing_format = None


def bench_hyb(
    g,
    x,
    y_golden,
    feat_size=128,
    bucket_sizes=[],
    coersening_factor=2,
    num_col_parts=1,
    use_implicit_unroll=False,
):
    num_buckets = len(bucket_sizes)
    coersening_factor = min(coersening_factor, feat_size // 32)
    indptr, indices, _ = g.adj_sparse("csc")
    m = g.num_dst_nodes()
    n = g.num_src_nodes()
    nnz = g.num_edges()
    global cached_bucketing_format
    if cached_bucketing_format is None:
        indptr_nd = tvm.nd.array(indptr.numpy(), device=tvm.cpu())
        indices_nd = tvm.nd.array(indices.numpy(), device=tvm.cpu())
        cached_bucketing_format = column_part_hyb(
            m, n, indptr_nd, indices_nd, num_col_parts, bucket_sizes
        )
    row_indices, col_indices, mask = cached_bucketing_format

    # rewrite csrmm
    nnz_cols_symbol = ell.params[-1]
    rewrites = []
    for part_id in range(num_col_parts):
        for bucket_id, bucket_size in enumerate(bucket_sizes):
            rewrites.append(
                FormatRewriteRule(
                    str(part_id) + "_" + str(bucket_id),
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
    mod = format_decompose(mod, rewrites)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    # specialize
    params = mod["main"].params
    param_map = {
        params[5]: m,  # m
        params[6]: n,  # n
        params[7]: feat_size // coersening_factor // 32,  # num_tiles,
        params[8]: nnz,  # nnz
        params[9]: coersening_factor,  # coersening_factor
    }
    for part_id in range(num_col_parts):
        for bucket_id in range(num_buckets):
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 4]] = m
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 5]] = n
            param_map[params[10 + 7 * (part_id * num_buckets + bucket_id) + 6]] = row_indices[
                part_id
            ][bucket_id].shape[0]

    mod["main"] = mod["main"].specialize(param_map).with_attr("horizontal_fuse", True)

    # schedule
    sch = tvm.tir.Schedule(mod)
    for sp_iter_name in [
        "csrmm_{}_{}".format(i, j) for j in range(num_buckets) for i in range(num_col_parts)
    ]:
        sp_iteration = sch.get_sparse_iteration(sp_iter_name)
        o, i, j, k1, k2, k3 = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [o, i])

    mod = sch.mod
    mod = tvm.sparse.lower_sparse_iter(mod)
    sch = tvm.tir.Schedule(mod)
    for part_id in range(num_col_parts):
        for bucket_id, bucket_size in enumerate(bucket_sizes):
            is_atomic = num_col_parts > 1 or bucket_id + 1 == num_buckets
            blk = sch.get_block("csrmm_{}_{}0".format(part_id, bucket_id))
            i, j, foo, foi, fi = sch.get_loops(blk)
            sch.reorder(foo, fi, j, foi)
            if is_atomic:
                sch.annotate(blk, "atomic", True)
                write_blk = sch.cache_write(blk, 0, "local")
                sch.reverse_compute_at(write_blk, fi, True)
                # sch.unroll(sch.get_loops(write_blk)[-2])
            sch.bind(fi, "threadIdx.x")
            sch.bind(foo, "blockIdx.y")
            sch.unroll(foi)
            if use_implicit_unroll:
                sch.annotate(foi, "pragma_unroll_explicit", 0)
            sch.unroll(j)
            if use_implicit_unroll:
                sch.annotate(j, "pragma_unroll_explicit", 0)
            io, ioi, ii = sch.split(i, [None, bucket_sizes[-1] // bucket_size, 8])
            sch.bind(io, "blockIdx.x")
            sch.bind(ii, "threadIdx.y")
            init_blk = sch.decompose_reduction(blk, fi)
            ax0, ax1 = sch.get_loops(init_blk)[-2:]
            sch.bind(ax0, "threadIdx.x")
            sch.unroll(ax1)
            if use_implicit_unroll:
                sch.annotate(ax1, "pragma_unroll_explicit", 0)

    mod = tvm.sparse.lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)
    f = tvm.build(mod, target="cuda")

    # prepare nd array
    b_nd = tvm.nd.array(
        x.numpy().reshape(-1).astype("float32"),
        device=tvm.cuda(0),
    )
    c_nd = tvm.nd.array(np.zeros((n * feat_size,)).astype("float32"), device=tvm.cuda(0))
    # prepare args
    args = [b_nd, c_nd]

    for part_id in range(num_col_parts):
        for bucket_id, _ in enumerate(bucket_sizes):
            weight = tvm.nd.array(
                mask[part_id][bucket_id].numpy().reshape(-1).astype("float32"), device=tvm.cuda(0)
            )
            rows = tvm.nd.array(
                row_indices[part_id][bucket_id].numpy().astype("int32"), device=tvm.cuda(0)
            )
            cols = tvm.nd.array(
                col_indices[part_id][bucket_id].numpy().reshape(-1).astype("int32"),
                device=tvm.cuda(0),
            )
            args += [weight, rows, cols]

    # test accuracy
    f(*args)
    tvm.testing.assert_allclose(c_nd.numpy().reshape(-1, feat_size), y_golden.numpy(), rtol=1e-4)

    # evaluate time
    evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=100)
    print("tir hyb time: {:.5f}ms".format(evaluator(*args).mean * 1000))


col_part_config = {
    "arxiv": 1,
    "proteins": 8,
    "pubmed": 1,
    "citeseer": 1,
    "cora": 1,
    "ppi": 16,
    "reddit": 8,
    "products": 16,
}

bucketing_config = {
    "arxiv": [1, 2, 4, 8, 16, 32],
    "proteins": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "pubmed": [1, 2, 4, 8, 16, 32],
    "citeseer": [1, 2, 4],
    "cora": [1, 2, 4],
    "ppi": [1, 2, 4, 8, 16, 32],
    "products": [1, 2, 4, 8, 16, 32],
    "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument("--dataset", "-d", type=str, default="arxiv", help="dataset name")
    parser.add_argument("--implicit-unroll", "-i", action="store_true", help="use implicit unroll")
    args = parser.parse_args()
    name = args.dataset
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
            coersening_factor=2,
            num_col_parts=col_part_config[name],
            use_implicit_unroll=args.implicit_unroll,
        )
