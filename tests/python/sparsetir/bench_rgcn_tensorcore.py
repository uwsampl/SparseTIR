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
from sparse_tir_composable_format_scripts import ell3d_fp16
from utils import get_hetero_dataset


def wmma_sync(d0: int, d1: int):
    @T.prim_func
    def wmma_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
        A_frag = T.match_buffer(
            a_frag, (d0, d1, 16), "float16", align=64, offset_factor=1, scope="wmma.matrix_a"
        )
        B_frag = T.match_buffer(
            b_frag, (16, 16), "float16", align=64, offset_factor=1, scope="wmma.matrix_b"
        )
        C_frag = T.match_buffer(
            c_frag, (d0, d1, 16), "float16", align=64, offset_factor=1, scope="wmma.accumulator"
        )

        with T.block("root"):
            for io, ii, j, k in T.grid(d0, d1, 16, 16):
                with T.block("update"):
                    vio, vii, vj, vk = T.axis.remap("SSSR", [io, ii, j, k])
                    T.block_attr({"sparse": True})
                    C_frag[vio, vii, vj] = (
                        C_frag[vio, vii, vj] + A_frag[vio, vii, vk] * B_frag[vk, vj]
                    )

    @T.prim_func
    def wmma_sync_16_1_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
        A_frag = T.match_buffer(
            a_frag, (16, 1, 16), "float16", align=64, offset_factor=1, scope="wmma.matrix_a"
        )
        B_frag = T.match_buffer(
            b_frag, (16, 16), "float16", align=64, offset_factor=1, scope="wmma.matrix_b"
        )
        C_frag = T.match_buffer(
            c_frag, (16, 1, 16), "float16", align=64, offset_factor=1, scope="wmma.accumulator"
        )

        with T.block("root"):
            for io, ii, j, k in T.grid(16, 1, 16, 16):
                with T.block("update"):
                    vio, vj, vk = T.axis.remap("SSR", [io, j, k])
                    T.block_attr({"sparse": True})
                    C_frag[vio, 0, vj] = (
                        C_frag[vio, 0, vj] + A_frag[vio, 0, vk] * B_frag[vk, vj]
                    )

    @T.prim_func
    def wmma_sync_1_16_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
        A_frag = T.match_buffer(
            a_frag, (1, 16, 16), "float16", align=64, offset_factor=1, scope="wmma.matrix_a"
        )
        B_frag = T.match_buffer(
            b_frag, (16, 16), "float16", align=64, offset_factor=1, scope="wmma.matrix_b"
        )
        C_frag = T.match_buffer(
            c_frag, (1, 16, 16), "float16", align=64, offset_factor=1, scope="wmma.accumulator"
        )

        with T.block("root"):
            for io, ii, j, k in T.grid(1, 16, 16, 16):
                with T.block("update"):
                    vii, vj, vk = T.axis.remap("SSR", [ii, j, k])
                    T.block_attr({"sparse": True})
                    C_frag[0, vii, vj] = (
                        C_frag[0, vii, vj] + A_frag[0, vii, vk] * B_frag[vk, vj]
                    )



    @T.prim_func
    def wmma_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
        A_frag = T.match_buffer(
            a_frag, (d0, d1, 16), "float16", align=64, offset_factor=16, scope="wmma.matrix_a"
        )
        B_frag = T.match_buffer(
            b_frag, (16, 16), "float16", align=64, offset_factor=16, scope="wmma.matrix_b"
        )
        C_frag = T.match_buffer(
            c_frag, (d0, d1, 16), "float16", align=64, offset_factor=16, scope="wmma.accumulator"
        )

        with T.block("root"):
            T.reads(
                [
                    C_frag[0:d0, 0:d1, 0:16],
                    A_frag[0:d0, 0:d1, 0:16],
                    B_frag[0:16, 0:16],
                ]
            )
            T.writes(C_frag[0:d0, 0:d1, 0:16])
            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                T.evaluate(
                    T.tvm_mma_sync(
                        C_frag.data,
                        C_frag.elem_offset // 256
                        + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                        A_frag.data,
                        A_frag.elem_offset // 256
                        + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                        B_frag.data,
                        B_frag.elem_offset // 256
                        + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                        C_frag.data,
                        C_frag.elem_offset // 256
                        + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                        dtype="handle",
                    )
                )

    if d0 == 1:
        return wmma_sync_1_16_desc, wmma_sync_impl
    elif d1 == 1:
        return wmma_sync_16_1_desc, wmma_sync_impl
    else:
        return wmma_sync_desc, wmma_sync_impl


def wmma_load_a(d0: int, d1: int, scope: str):
    @T.prim_func
    def wmma_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
        A = T.match_buffer(a, (d0, d1, 16), "float16", align=64, offset_factor=16, scope=scope)
        A_frag = T.match_buffer(
            a_frag, (d0, d1, 16), "float16", align=64, offset_factor=16, scope="wmma.matrix_a"
        )

        with T.block("root"):
            for io, ii, j in T.grid(d0, d1, 16):
                with T.block("load"):
                    vio, vii, vj = T.axis.remap("SSS", [io, ii, j])
                    A_frag[vio, vii, vj] = A[vio, vii, vj]
    
    @T.prim_func
    def wmma_load_a_16_1_desc(a: T.handle, a_frag: T.handle) -> None:
        A = T.match_buffer(a, (16, 1, 16), "float16", align=64, offset_factor=16, scope=scope)
        A_frag = T.match_buffer(
            a_frag, (16, 1, 16), "float16", align=64, offset_factor=16, scope="wmma.matrix_a"
        )

        with T.block("root"):
            for io, ii, j in T.grid(16, 1, 16):
                with T.block("load"):
                    vio, vj = T.axis.remap("SS", [io, j])
                    A_frag[vio, 0, vj] = A[vio, 0, vj]
    
    @T.prim_func
    def wmma_load_a_1_16_desc(a: T.handle, a_frag: T.handle) -> None:
        A = T.match_buffer(a, (1, 16, 16), "float16", align=64, offset_factor=16, scope=scope)
        A_frag = T.match_buffer(
            a_frag, (1, 16, 16), "float16", align=64, offset_factor=16, scope="wmma.matrix_a"
        )

        with T.block("root"):
            for io, ii, j in T.grid(1, 16, 16):
                with T.block("load"):
                    vii, vj = T.axis.remap("SS", [ii, j])
                    A_frag[0, vii, vj] = A[0, vii, vj]
 

    @T.prim_func
    def wmma_load_a_impl(a: T.handle, a_frag: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        s2 = T.var("int32")
        A = T.match_buffer(
            a,
            (d0, d1, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope=scope,
            strides=[s0, s1, s2],
        )
        A_frag = T.match_buffer(
            a_frag, (d0, d1, 16), "float16", align=64, offset_factor=16, scope="wmma.matrix_a"
        )

        with T.block("root"):
            T.reads(A[0:d0, 0:d1, 0:16])
            T.writes(A_frag[0:d0, 0:d1, 0:16])
            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                T.evaluate(
                    T.tvm_load_matrix_sync(
                        A_frag.data,
                        16,
                        16,
                        16,
                        A_frag.elem_offset // 256
                        + T.floordiv(T.floormod(A_frag.elem_offset, 256), 16),
                        A.access_ptr("r"),
                        s1,
                        "row_major",
                        dtype="handle",
                    )
                )

    if d0 == 1:
        return wmma_load_a_1_16_desc, wmma_load_a_impl
    elif d1 == 1:
        return wmma_load_a_16_1_desc, wmma_load_a_impl
    else:
        return wmma_load_a_desc, wmma_load_a_impl


def wmma_load_b(scope: str):
    @T.prim_func
    def wmma_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
        B = T.match_buffer(b, (16, 16), "float16", align=64, offset_factor=16, scope=scope)
        B_frag = T.match_buffer(
            b_frag, (16, 16), "float16", align=64, offset_factor=16, scope="wmma.matrix_b"
        )
        with T.block("root"):
            for i, j in T.grid(16, 16):
                with T.block("load"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B_frag[vi, vj] = B[vi, vj]

    @T.prim_func
    def wmma_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        B = T.match_buffer(
            b, (16, 16), "float16", align=64, offset_factor=16, scope=scope, strides=[s0, s1]
        )
        B_frag = T.match_buffer(
            b_frag, (16, 16), "float16", align=64, offset_factor=16, scope="wmma.matrix_b"
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
                        B_frag.elem_offset // 256
                        + T.floordiv(T.floormod(B_frag.elem_offset, 256), 16),
                        B.access_ptr("r"),
                        B.strides[0],
                        "row_major",
                        dtype="handle",
                    )
                )

    return wmma_load_b_desc, wmma_load_b_impl


def wmma_fill(d0: int, d1: int):
    @T.prim_func
    def wmma_fill_desc(c_frag: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag, (d0, d1, 16), "float16", align=64, offset_factor=16, scope="wmma.accumulator"
        )
        with T.block("root"):
            for io, ii, j in T.grid(d0, d1, 16):
                with T.block("init"):
                    vio, vii, vj = T.axis.remap("SSS", [io, ii, j])
                    C_frag[vio, vii, vj] = T.float16(0)

    @T.prim_func
    def wmma_fill_16_1_desc(c_frag: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag, (16, 1, 16), "float16", align=64, offset_factor=16, scope="wmma.accumulator"
        )
        with T.block("root"):
            for io, ii, j in T.grid(16, 1, 16):
                with T.block("init"):
                    vio, vj = T.axis.remap("SS", [io, j])
                    C_frag[vio, 0, vj] = T.float16(0)

    @T.prim_func
    def wmma_fill_1_16_desc(c_frag: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag, (1, 16, 16), "float16", align=64, offset_factor=16, scope="wmma.accumulator"
        )
        with T.block("root"):
            for io, ii, j in T.grid(1, 16, 16):
                with T.block("init"):
                    vii, vj = T.axis.remap("SS", [ii, j])
                    C_frag[0, vii, vj] = T.float16(0)


    @T.prim_func
    def wmma_fill_impl(c_frag: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag, (d0, d1, 16), "float16", align=64, offset_factor=16, scope="wmma.accumulator"
        )
        with T.block("root"):
            T.reads([])
            T.writes(C_frag[0:d0, 0:d1, 0:16])
            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                T.evaluate(
                    T.tvm_fill_fragment(
                        C_frag.data,
                        16,
                        16,
                        16,
                        C_frag.elem_offset // 256
                        + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                        T.float16(0),
                        dtype="handle",
                    )
                )

    if d0 == 1:
        return wmma_fill_1_16_desc, wmma_fill_impl
    elif d1 == 1:
        return wmma_fill_16_1_desc, wmma_fill_impl
    else:
        return wmma_fill_desc, wmma_fill_impl


def wmma_store(d0: int, d1: int, scope: str):
    @T.prim_func
    def wmma_store_desc(c_frag: T.handle, c: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag, (d0, d1, 16), "float16", align=64, offset_factor=16, scope="wmma.accumulator"
        )
        C = T.match_buffer(c, (d0, d1, 16), "float16", align=64, offset_factor=16, scope=scope)
        with T.block("root"):
            for io, ii, j in T.grid(d0, d1, 16):
                with T.block("store"):
                    vio, vii, vj = T.axis.remap("SSS", [io, ii, j])
                    C[vio, vii, vj] = C_frag[vio, vii, vj]

    @T.prim_func
    def wmma_store_desc_16_1(c_frag: T.handle, c: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag, (16, 1, 16), "float16", align=64, offset_factor=16, scope="wmma.accumulator"
        )
        C = T.match_buffer(c, (16, 1, 16), "float16", align=64, offset_factor=16, scope=scope)
        with T.block("root"):
            for io, ii, j in T.grid(16, 1, 16):
                with T.block("store"):
                    vio, vj = T.axis.remap("SS", [io, j])
                    C[vio, 0, vj] = C_frag[vio, 0, vj]
    
    @T.prim_func
    def wmma_store_desc_1_16(c_frag: T.handle, c: T.handle) -> None:
        C_frag = T.match_buffer(
            c_frag, (1, 16, 16), "float16", align=64, offset_factor=16, scope="wmma.accumulator"
        )
        C = T.match_buffer(c, (1, 16, 16), "float16", align=64, offset_factor=16, scope=scope)
        with T.block("root"):
            for io, ii, j in T.grid(1, 16, 16):
                with T.block("store"):
                    vii, vj = T.axis.remap("SS", [ii, j])
                    C[0, vii, vj] = C_frag[0, vii, vj]
 

    @T.prim_func
    def wmma_store_impl(c_frag: T.handle, c: T.handle) -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        s2 = T.var("int32")
        C_frag = T.match_buffer(
            c_frag, (d0, d1, 16), "float16", align=64, offset_factor=16, scope="wmma.accumulator"
        )
        C = T.match_buffer(
            c,
            (d0, d1, 16),
            "float16",
            align=64,
            offset_factor=16,
            scope=scope,
            strides=[s0, s1, s2],
        )
        with T.block("root"):
            T.reads(C_frag[0:d0, 0:d1, 0:16])
            T.writes(C[0:d0, 0:d1, 0:16])
            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                T.evaluate(
                    T.tvm_store_matrix_sync(
                        C_frag.data,
                        16,
                        16,
                        16,
                        C_frag.elem_offset // 256
                        + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                        C.access_ptr("w"),
                        s1,
                        "row_major",
                        dtype="handle",
                    )
                )

    if d0 == 1:
        return wmma_store_desc_1_16, wmma_store_impl 
    elif d1 == 1:
        return wmma_store_desc_16_1, wmma_store_impl
    else:
        return wmma_store_desc, wmma_store_impl


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
    W = T.match_sparse_buffer(w, (R, F_in, F_out), "float16")
    X = T.match_sparse_buffer(x, (J_detach, F_in), "float16")
    Y = T.match_sparse_buffer(y, (I_detach, F_out), "float16")
    WX = T.match_sparse_buffer(wx, (R, I, J, F_out), "float16")

    with T.iter([R, I, J, F_out, F_in], "SSSSR", "rgcn-hetero-forward_wx") as [r, i, j, fo, fi]:
        with T.init():
            WX[r, i, j, fo] = T.float16(0)
        WX[r, i, j, fo] += X[j, fi] * W[r, fi, fo]

    with T.iter([R, I, J, F_out], "SSRS", "rgcn-hetero-forward") as [r, i, j, fo]:
        with T.init():
            Y[i, fo] = T.float16(0)
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
                        feat_size_sym: feat_size,
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
    # register load_b
    tir.TensorIntrin.register("wmma_{}_load_b".format("shared"), *wmma_load_b("shared"))
    tir.TensorIntrin.register("wmma_{}_load_b".format("global"), *wmma_load_b("global"))

    for bucket_id, bucket_size in enumerate(buckets):
        d0 = group_size // bucket_size
        d1 = bucket_size
        tir.TensorIntrin.register(
            "wmma_{}_{}_{}_store".format(d0, d1, "shared"), *wmma_store(d0, d1, "shared")
        )
        tir.TensorIntrin.register(
            "wmma_{}_{}_{}_load_a".format(d0, d1, "shared"), *wmma_load_a(d0, d1, "shared")
        )
        tir.TensorIntrin.register("wmma_{}_{}_init".format(d0, d1), *wmma_fill(d0, d1))
        tir.TensorIntrin.register("wmma_{}_{}_sync".format(d0, d1), *wmma_sync(d0, d1))
        blk_wx = sch.get_block("rgcn-hetero-forward_wx_{}0".format(bucket_id))
        blk = sch.get_block("rgcn-hetero-forward_{}0".format(bucket_id))
        sch.match_to_alloc(blk_wx, 0)
        sch.set_scope(blk_wx, 0, "shared")
        sch.compute_at(blk_wx, sch.get_loops(blk)[0], True)
        W_shared = sch.reverse_cache_read(blk_wx, 2, "shared", [0, 1, 5, 4])
        sch.compute_at(W_shared, sch.get_loops(blk)[0], True)
        ax0, ax1, ax2, ax3 = sch.get_loops(blk_wx)[-4:]
        ax2_o, ax2_i = sch.split(ax2, [None, 16])
        ax3_o, ax3_i = sch.split(ax3, [None, 16])
        sch.reorder(ax2_o, ax3_o, ax0, ax1, ax2_i, ax3_i)
        X_shared = sch.reverse_cache_read(blk_wx, 0, "shared")
        sch.compute_at(X_shared, ax3_o, True)
        WX_accum = sch.reverse_cache_write(blk_wx, 0, "wmma.accumulator")
        W_wmma = sch.reverse_cache_read(blk_wx, 2, "wmma.matrix_b", [0, 1, 5, 4])
        # W_wmma = sch.cache_read(blk_wx, 2, "wmma.matrix_b")
        sch.compute_at(W_wmma, ax3_o, True)
        X_wmma = sch.reverse_cache_read(blk_wx, 0, "wmma.matrix_a")
        sch.bind(sch.get_loops(blk)[0], "blockIdx.x")
        sch.decompose_reduction(blk_wx, ax3_o)

        # unroll
        ax2 = sch.get_loops(WX_accum)[-4]
        sch.unroll(ax2)
        ax5 = sch.get_loops(blk_wx)[-5]
        ax4 = sch.get_loops(blk_wx)[-6]
        sch.unroll(ax5)
        sch.unroll(ax4)

        # tensorize
        sch.tensorize(sch.get_loops(WX_accum)[-3], "wmma_{}_{}_{}_store".format(d0, d1, "shared"))
        print(sch.mod["main"].script())
        sch.tensorize(sch.get_loops(X_wmma)[-3], "wmma_{}_{}_{}_load_a".format(d0, d1, "shared"))
        sch.tensorize(sch.get_loops(W_wmma)[-2], "wmma_{}_load_b".format("shared"))
        sch.tensorize(
            sch.get_loops(sch.get_block("rgcn-hetero-forward_wx_{}0_init".format(bucket_id)))[-3],
            "wmma_{}_{}_init".format(d0, d1),
        )
        sch.hide_buffer_access(blk_wx, "read", [2, 4])
        sch.tensorize(sch.get_loops(blk_wx)[-4], "wmma_{}_{}_sync".format(d0, d1))

        # schedule W_shared
        ax2, ax3 = sch.get_loops(W_shared)[-2:]
        fused_ax = sch.fuse(ax2, ax3)
        ax0, ax1, ax2 = sch.split(fused_ax, [None, 32, 8])
        sch.vectorize(ax2)
        sch.bind(ax1, "threadIdx.x")
        sch.unroll(ax0)

        # schedule X_shared
        ax0, ax1, ax2 = sch.get_loops(X_shared)[-3:]
        fused_ax = sch.fuse(ax0, ax1, ax2)
        ax0, ax1, ax2 = sch.split(fused_ax, [None, 32, 8])
        sch.vectorize(ax2)
        sch.bind(ax1, "threadIdx.x")

        # schedule for the write block
        ii, j, fo = sch.get_loops(blk)[-3:]
        sch.annotate(blk, "atomic", True)
        Y_local = sch.reverse_cache_write(blk, 0, "local")
        sch.reverse_compute_at(Y_local, ii, True)
        sch.bind(fo, "threadIdx.x")
        sch.unroll(j)
        sch.unroll(ii)
        sch.bind(sch.get_loops(Y_local)[-1], "threadIdx.x")

    mod = lower_sparse_buffer(sch.mod)
    mod = tvm.tir.transform.RemoveUnusedArgs()(mod)

    f = tvm.build(mod["main"], target="cuda")
    print(f.imported_modules[0].get_source())

    # prepare inputs
    dev = tvm.cuda(0)
    W_nd = tvm.nd.array(weight.transpose(-1, -2).contiguous().half().cpu().view(-1), device=dev)
    X_nd = tvm.nd.array(feat.half().cpu().view(-1), device=dev)
    Y_nd = tvm.nd.array(th.zeros(m * feat_size, dtype=th.float16), device=dev)
    args = [W_nd, X_nd, Y_nd]
    for bucket_id, _ in enumerate(buckets):
        args.append(
            tvm.nd.array(mask[bucket_id].numpy().reshape(-1).astype(np.float16), device=dev)
        )
        args.append(tvm.nd.array(row_indices[bucket_id].numpy().reshape(-1), device=dev))
        args.append(tvm.nd.array(col_indices[bucket_id].numpy().reshape(-1), device=dev))
    for bucket_id, _ in enumerate(buckets):
        args.append(tvm.nd.array(mids[bucket_id], device=dev))
    f(*args)

    # tvm.testing.assert_allclose(Y_nd.numpy(), ground_truth_y.half().cpu().numpy().flatten(), rtol=1e-1)

    # evaluate time
    evaluator = f.time_evaluator(f.entry_name, tvm.cuda(0), number=10)
    print("sparse-tir:\t\t {:.3f} ms".format(evaluator(*args).mean * 1000))


if __name__ == "__main__":
    for feat_size in [32]:  # [4, 8, 16, 32]:
        for name in ["biokg"]:  # ["aifb", "mutag", "bgs", "am", "biokg"]:
            print("dataset {}, feat_size={}:".format(name, feat_size))
            dataset = get_hetero_dataset(name)
            g = dataset[0]
            type_pointers = prepare_hetero_graph_simplified(g)
            n = g.num_nodes()
            r = len(g.etypes)
            feat = th.rand(n, feat_size).to(0) / 100
            weight = th.rand(r, feat_size, feat_size).to(0) / 100
            # homograph
            ground_truth_y = get_ground_truth(g, type_pointers, feat, weight)
            test_rgcn_composable_format(
                g, type_pointers, feat_size, feat, weight, ground_truth_y, 4, 16, [4]
            )
