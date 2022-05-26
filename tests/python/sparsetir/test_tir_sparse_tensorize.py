import tvm
from tvm import tir
from tvm.script import tir as T
import tvm.testing
import numpy as np
import scipy.sparse as sp
from tvm.ir import IRModule
from tqdm import tqdm
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer


@T.prim_func
def bsrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    nnzb: T.int32,
    blk: T.int32,
    feat_size: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(nb)
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), "int32")
    J_detach = T.dense_fixed(mb)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float16")
    B = T.match_sparse_buffer(b, (J_detach, BJ, F), "float16")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([I, BI, BJ, F, J], "SSRSR", "bsrmm") as [
        i,
        bi,
        bj,
        f,
        j,
    ]:
        with T.init():
            C[i, bi, f] = 0.0
        C[i, bi, f] = C[i, bi, f] + T.float32(A[i, j, bi, bj]) * T.float32(B[j, bj, f])


@T.prim_func
def wmma_sync_desc(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(a_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a")
    B_frag = T.match_buffer(b_frag, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b")
    C_frag = T.match_buffer(c_frag, (16, 16), "float32", align=128, offset_factor=1, scope="wmma.accumulator")

    with T.block("root"):
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                T.block_attr({"sparse": True})
                C_frag[vii, vjj] = C_frag[vii, vjj] + T.cast(A_frag[vii, vkk], "float32") * T.cast(
                    B_frag[vkk, vjj], "float32"
                )


@T.prim_func
def wmma_sync_impl(a_frag: T.handle, b_frag: T.handle, c_frag: T.handle) -> None:
    A_frag = T.match_buffer(a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")
    B_frag = T.match_buffer(b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
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
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
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
def wmma_load_a_desc(a: T.handle, a_frag: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    A_frag = T.match_buffer(a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(A_frag[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                A_frag[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_a_impl(a: T.handle, a_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    A_frag = T.match_buffer(a_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(A_frag[0:16, 0:16])
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
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
def wmma_load_b_desc(b: T.handle, b_frag: T.handle) -> None:
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    B_frag = T.match_buffer(b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                B_frag[vii, vjj] = B[vii, vjj]


@T.prim_func
def wmma_load_b_impl(b: T.handle, b_frag: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    B = T.match_buffer(
        b, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    B_frag = T.match_buffer(b_frag, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    with T.block("root"):
        T.reads(B[0:16, 0:16])
        T.writes(B_frag[0:16, 0:16])
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
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
def wmma_fill_desc(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C_frag[vii, vjj] = T.float32(0)


@T.prim_func
def wmma_fill_impl(c_frag: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        T.reads([])
        T.writes(C_frag[0:16, 0:16])
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
        T.evaluate(
            T.tvm_fill_fragment(
                C_frag.data,
                16,
                16,
                16,
                C_frag.elem_offset // 256 + T.floordiv(T.floormod(C_frag.elem_offset, 256), 16),
                T.float32(0),
                dtype="handle",
            )
        )


@T.prim_func
def wmma_store_desc(c_frag: T.handle, c: T.handle) -> None:
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = C_frag[vii, vjj]


@T.prim_func
def wmma_store_impl(c_frag: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    C_frag = T.match_buffer(
        c_frag, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (16, 16), "float32", align=128, offset_factor=16, scope="global", strides=[s0, s1]
    )
    with T.block("root"):
        T.reads(C_frag[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
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


WMMA_SYNC = tir.TensorIntrin.register(
    "wmma_sync",
    wmma_sync_desc,
    wmma_sync_impl,
)

WMMA_LOAD_A = tir.TensorIntrin.register(
    "wmma_load_a",
    wmma_load_a_desc,
    wmma_load_a_impl,
)

WMMA_LOAD_B = tir.TensorIntrin.register(
    "wmma_load_b",
    wmma_load_b_desc,
    wmma_load_b_impl,
)

WMMA_FILL = tir.TensorIntrin.register(
    "wmma_fill",
    wmma_fill_desc,
    wmma_fill_impl,
)


WMMA_STORE = tir.TensorIntrin.register(
    "wmma_store",
    wmma_store_desc,
    wmma_store_impl,
)


block_size = 16
nb = 32
mb = 32
feat_size = 256
n = nb * block_size
m = mb * block_size

A_block = sp.random(mb, nb, dtype="float32", density=0.05, format="csr", random_state=0)
indptr = A_block.indptr
indices = A_block.indices
nnzb = A_block.nnz
np.random.seed(0)
data = np.random.rand(nnzb, block_size, block_size)
A = sp.bsr_matrix((data.astype("float16"), indices, indptr), shape=(n, m))
x = np.random.rand(m, feat_size).astype("float16")
y_ground_truth = A * x
y = np.zeros((n * feat_size,)).astype("float32")

v_nb, v_mb, v_nnzb, v_blk, v_feat_size = bsrmm.params[-5:]
bsrmm = bsrmm.specialize(
    {v_nb: nb, v_mb: mb, v_nnzb: nnzb, v_blk: block_size, v_feat_size: feat_size}
)
sch = tvm.tir.Schedule(bsrmm)
sp_iteration = sch.get_sparse_iteration("bsrmm")
i, bi, bj, f, j = sch.get_sp_iters(sp_iteration)
sch.sparse_reorder(sp_iteration, [i, j, bi, f, bj])
mod = lower_sparse_iter(sch.mod)
sch = tir.Schedule(mod)
blk_inner = sch.get_block("bsrmm1")
blk_outer = sch.get_block("bsrmm0")
print(sch.mod["main"].script(), sch.get_loops(blk_inner))
j, bi, f, bj = sch.get_loops(blk_inner)
fo, fi = sch.split(f, [None, 16])
sch.reorder(fo, j, bi, fi, bj)
i, = sch.get_loops(blk_outer)
sch.bind(i, "blockIdx.x")
sch.bind(fo, "blockIdx.y")
# sch.lift_loop(fo)
new_blk = sch.blockize(bi)
C_local = sch.cache_write(new_blk, 0, "wmma.accumulator")
sch.reverse_compute_at(C_local, fo)
sch.decompose_reduction(new_blk, j)
A_local = sch.cache_read(blk_inner, 1, "wmma.matrix_a")
B_local = sch.cache_read(blk_inner, 2, "wmma.matrix_b")
sch.hide_buffer_access(blk_inner, "read", [3])
sch.tensorize(sch.get_loops(blk_inner)[-3], "wmma_sync")
sch.tensorize(sch.get_loops(B_local)[-2], "wmma_load_b")
sch.tensorize(sch.get_loops(A_local)[-2], "wmma_load_a")
sch.tensorize(sch.get_loops(C_local)[-2], "wmma_store")
sch.tensorize(sch.get_loops(sch.get_block("bsrmm1_init"))[-2], "wmma_fill")
mod = lower_sparse_buffer(sch.mod)
print(mod["main"].script())


# for t in tqdm(range(0, 2)):
#     f = tvm.build(mod["main"], target="cuda")
#     ctx = tvm.cuda(0)
#     A_indptr = tvm.nd.array(np.copy(indptr).astype("int32"), device=ctx)
#     A_indices = tvm.nd.array(np.copy(indices).astype("int32"), device=ctx)
#     A_data = tvm.nd.array(np.copy(data).astype("float16"), device=ctx)
#     X_nd = tvm.nd.array(np.copy(x.reshape(mb, block_size, feat_size)).astype("float16"), device=ctx)
#     Y_nd = tvm.nd.array(np.zeros((nb, block_size, feat_size), dtype="float32"), device=ctx)
#     f(A_data, X_nd, Y_nd, A_indptr, A_indices)
#     tvm.testing.assert_allclose(
#         np.copy(y_ground_truth).reshape(nb, block_size, feat_size),
#         Y_nd.numpy(),
#         rtol=1e-5,
#         atol=1e-5,
#     )
# evaluator = f.time_evaluator(f.entry_name, ctx, number=10)
# print("w/o Tensor Cores:")
# print(evaluator(A_data, X_nd, Y_nd, A_indptr, A_indices))

# for t in tqdm(range(0, 2)):
for t in range(2):
    f = tvm.build(mod["main"], target="cuda")
    print(f.imported_modules[0].get_source())
    ctx = tvm.cuda(0)
    A_indptr = tvm.nd.array(np.copy(indptr).astype("int32"), device=ctx)
    A_indices = tvm.nd.array(np.copy(indices).astype("int32"), device=ctx)
    A_data = tvm.nd.array(np.copy(data).reshape(-1).astype("float16"), device=ctx)
    X_nd = tvm.nd.array(np.copy(x.reshape(-1)).astype("float16"), device=ctx)
    Y_nd = tvm.nd.array(np.zeros((nb * block_size * feat_size), dtype="float32"), device=ctx)
    print(A_data)
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(
        np.copy(y_ground_truth).reshape(-1),
        Y_nd.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
print("with Tensor Cores:")
evaluator = f.time_evaluator(f.entry_name, ctx, number=10)
print(evaluator(A_data, X_nd, Y_nd, A_indptr, A_indices))
