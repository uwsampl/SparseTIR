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
# from sparse_tir_scripts import rgcn_hetero_forward
from tvm.sparse import (
    lower_sparse_iter,
    lower_sparse_buffer,
    FormatRewriteRule,
    format_decompose,
    csf_to_ell3d,
)
from typing import List, Tuple, Mapping
from sparse_tir_composable_format_scripts import ell3d_fp16
from utils import get_hetero_dataset

@T.prim_func
def wmma_m16n16k16_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.accumulator"
    )

    with T.block("root"):
        for i, k, j in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vkk, vjj = T.axis.remap("SRS", [i, k, j])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = C_frag[vii, vjj] + A_frag[vii, vkk] * B_frag[vkk, vjj]


@T.prim_func
def wmma_m16n16k16_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )

    with T.block("root"):
        T.reads(
            [
                C_frag[0:16, 0:16],
                A_frag[0:16, 0:16],
                B_frag[0:16, 0:16],
            ]
        )
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_mma_sync(
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    A_frag.data,
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    B_frag.data,
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    C_frag.data,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    dtype="handle",
                )
            )

@T.prim_func
def wmma_m16n16k16_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(A_frag[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                A_frag[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_m16n16k16_load_a_impl(a: T.handle, a_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    A_frag = T.match_buffer(
        a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(A_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_load_matrix_sync(
                    A_frag.data,
                    16,
                    16,
                    16,
                    A_frag.elem_offset // 256 + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                    A.access_ptr("r"),
                    A.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )

@T.prim_func
def wmma_m16n16k16_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16, scope="shared")
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vii, vjj]


@T.prim_func
def wmma_m16n16k16_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(
        b, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s0, s1]
    )
    B_frag = T.match_buffer(
        b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    with T.block("root"):
        T.reads(B[0:16, 0:16])
        T.writes(B_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_load_matrix_sync(
                    B_frag.data,
                    16,
                    16,
                    16,
                    B_frag.elem_offset // 256 + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                    B.access_ptr("r"),
                    B.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )

@T.prim_func
def wmma_m16n16k16_fill_desc(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float16(0)


@T.prim_func
def wmma_m16n16k16_fill_impl(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        T.reads([])
        T.writes(C_frag[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_fill_fragment(
                    C_frag.data,
                    16,
                    16,
                    16,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    T.float16(0),
                    dtype="handle",
                )
            )

@T.prim_func
def wmma_m16n16k16_store_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_m16n16k16_store_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    with T.block("root"):
        T.reads(C_frag[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for tx in T.thread_binding(0, 32, "threadIdx.x"):
            T.evaluate(
                T.tvm_store_matrix_sync(
                    C_frag.data,
                    16,
                    16,
                    16,
                    C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                    C.access_ptr("w"),
                    C.strides[0],
                    "row_major",
                    dtype="handle",
                )
            )

WMMA_M16N16K16_SYNC = tir.TensorIntrin.register(
    "wmma_m16n16k16_sync",
    wmma_m16n16k16_sync_desc,
    wmma_m16n16k16_sync_impl,
)

WMMA_M16N16K16_LOAD_A = tir.TensorIntrin.register(
    "wmma_m16n16k16_load_a",
    wmma_m16n16k16_load_a_desc,
    wmma_m16n16k16_load_a_impl,
)

WMMA_M16N16K16_LOAD_B = tir.TensorIntrin.register(
    "wmma_m16n16k16_load_b",
    wmma_m16n16k16_load_b_desc,
    wmma_m16n16k16_load_b_impl,
)

WMMA_M16N16K16_FILL = tir.TensorIntrin.register(
    "wmma_m16n16k16_fill",
    wmma_m16n16k16_fill_desc,
    wmma_m16n16k16_fill_impl,
)

WMMA_M16N16K16_STORE = tir.TensorIntrin.register(
    "wmma_m16n16k16_store",
    wmma_m16n16k16_store_desc,
    wmma_m16n16k16_store_impl,
)



@T.prim_func
def rgcn_hetero_forward(
    a: T.handle,
    w: T.handle,
    x: T.handle,
    y: T.handle,
    wx: T.handle,
    indptr_i: T.handle,
    indices_i: T.handle,
    indptr_j: T.handle,
    indices_j: T.handle,
    m: T.int32,
    n: T.int32,
    num_rels: T.int32,
    feat_size: T.int32,
    nnz_i: T.int32,
    nnz_j: T.int32,
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    R = T.dense_fixed(num_rels)
    I = T.sparse_variable(R, (m, nnz_i), (indptr_i, indices_i), "int32")
    J = T.sparse_variable(I, (n, nnz_j), (indptr_j, indices_j), "int32")
    I_detach = T.dense_fixed(m)
    J_detach = T.dense_fixed(n)
    F_in = T.dense_fixed(feat_size)
    F_out = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (R, I, J), "float16")
    W = T.match_sparse_buffer(w, (R, F_out, F_in), "float16")
    X = T.match_sparse_buffer(x, (J_detach, F_in), "float16")
    Y = T.match_sparse_buffer(y, (I_detach, F_out), "float16")
    WX = T.match_sparse_buffer(wx, (R, I, J, F_out), "float16")

    with T.iter([R, I, J, F_out, F_in], "SSSSR", "rgcn-hetero-forward_wx") as [r, i, j, fo, fi]:
        with T.init():
            WX[r, i, j, fo] = T.float16(0)
        WX[r, i, j, fo] += W[r, fo, fi] * X[j, fi]

    with T.iter([R, I, J, F_out], "SSRS", "rgcn-hetero-forward") as [r, i, j, fo]:
        with T.init():
            Y[i, fo] = 0.0
        Y[i, fo] = Y[i, fo] + A[r, i, j] * WX[r, i, j, fo]




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


def csf_to_ell3d_inv_idx_map(r, io, ii, j, fo):
    return r, ii, j, fo


def csf_to_ell3d_idx_map(r, i, j, fo):
    return r, 0, i, j, fo


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

    d0, d1, d2, nnz, nnz_rows, nnz_cols, feat_size_sym = ell3d_fp16.params[-7:]
    rewrites = []
    for bucket_id, bucket_size in enumerate(buckets):
        rewrites.append(
            FormatRewriteRule(
                str(bucket_id),
                ell3d_fp16.specialize(
                    {
                        d0: num_rels,
                        d1: m,
                        d2: n,
                        nnz: row_indices[bucket_id].shape[0],
                        nnz_rows: group_size // bucket_size,
                        nnz_cols: bucket_size,
                        feat_size_sym: feat_size
                    }
                ),
                ["A", "WX"],
                ["R", "I", "J", "F_out"],
                ["R", "IO", "II", "J", "FO"],
                {"R": ["R"], "I": ["IO", "II"], "J": ["J"], "F_out": ["FO"]},
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
    mod = format_decompose(mod, rewrites, include_format_rewrite_blks=False)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    sch = tir.Schedule(mod["main"])
    for bucket_id, _ in enumerate(buckets):
        sp_iteration = sch.get_sparse_iteration("rgcn-hetero-forward_wx_{}".format(bucket_id))
        r, io, ii, j, fo, fi = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [r, io])
        sp_iteration = sch.get_sparse_iteration("rgcn-hetero-forward_{}".format(bucket_id))
        r, io, ii, j, fo = sch.get_sp_iters(sp_iteration)
        sch.sparse_fuse(sp_iteration, [r, io])
    mod = lower_sparse_iter(sch.mod)
    mod = tvm.tir.transform.RemovePreprocess()(mod)

    sch = tir.Schedule(mod["main"])
    for bucket_id, bucket_size in enumerate(buckets):
        blk_wx = sch.get_block("rgcn-hetero-forward_wx_{}0".format(bucket_id))
        blk = sch.get_block("rgcn-hetero-forward_{}0".format(bucket_id))
        sch.match_to_alloc(blk_wx, 0)
        sch.set_scope(blk_wx, 0, "shared")
        sch.compute_at(blk_wx, sch.get_loops(blk)[0])
        W_shared = sch.reverse_cache_read(blk_wx, 0, "shared")
        sch.compute_at(W_shared, sch.get_loops(blk)[0])
        ax0, ax1, ax2, ax3 = sch.get_loops(blk_wx)[-4:]
        fused_ax = sch.fuse(ax0, ax1)
        ax2_o, ax2_i = sch.split(ax2, [None, 16])
        ax3_o, ax3_i = sch.split(ax3, [None, 16])
        sch.reorder(ax2_o, ax3_o, fused_ax, ax2_i, ax3_i)
        X_shared = sch.reverse_cache_read(blk_wx, 2, "shared")
        sch.compute_at(X_shared, ax3_o)
        WX_accum = sch.reverse_cache_write(blk_wx, 0, "wmma.accumulator")
        
        print(sch.mod["main"].script())
        # io, ii, j, fo, fi = sch.get_loops(blk)
        # sch.rfactor(j, 1)
        # print(sch.mod["main"].script())
    """
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
    """


if __name__ == "__main__":
    for feat_size in [32]:  # [4, 8, 16, 32]:
        for name in ["aifb"]: #["aifb", "mutag", "bgs", "am", "biokg"]:
            print("dataset {}, feat_size={}:".format(name, feat_size))
            dataset = get_hetero_dataset(name)
            g = dataset[0]
            type_pointers = prepare_hetero_graph_simplified(g)
            n = g.num_nodes()
            r = len(g.etypes)
            feat = th.rand(n, feat_size).to(0) / 100
            weight = th.rand(r, feat_size, feat_size).to(0)
            # homograph
            ground_truth_y = get_ground_truth(g, type_pointers, feat, weight)
            test_rgcn_composable_format(
                g, type_pointers, feat_size, feat, weight, ground_truth_y, 4, 16, [8]
            )
