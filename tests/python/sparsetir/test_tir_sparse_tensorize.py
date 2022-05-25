import tvm
from tvm import tir
from tvm.script import tir as T
import tvm.testing
import numpy as np
import scipy.sparse as sp
from tvm.ir import IRModule
from tqdm import tqdm


@T.prim_func
def wmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=1, scope="wmma.accumulator")

    with T.block("root"):
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                T.block_attr({"sparse": True})
                C[vii, vjj] = C[vii, vjj] + T.cast(A[vii, vkk], "float32") * T.cast(
                    B[vkk, vjj], "float32"
                )


@T.prim_func
def wmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    C = T.match_buffer(
        c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )

    with T.block("root"):
        T.reads(
            [
                C[0:16, 0:16],
                A[0:16, 0:16],
                B[0:16, 0:16],
            ]
        )
        T.writes(C[0:16, 0:16])
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
        T.evaluate(
            T.tvm_mma_sync(
                C.data,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.data,
                A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                B.data,
                B.elem_offset // 256 + T.floordiv(T.floormod(B.elem_offset, 256), 16),
                C.data,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                dtype="handle",
            )
        )


@T.prim_func
def wmma_load_a_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")

    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s1, s0]
    )
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
        T.evaluate(
            T.tvm_load_matrix_sync(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.access_ptr("r"),
                s1,
                "row_major",
                dtype="handle",
            )
        )


@T.prim_func
def wmma_load_b_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="global")
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_b_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="global", strides=[s1, s0]
    )
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
        T.evaluate(
            T.tvm_load_matrix_sync(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.access_ptr("r"),
                s1,
                "row_major",
                dtype="handle",
            )
        )


@T.prim_func
def wmma_fill_desc(c: T.handle) -> None:
    C = T.match_buffer(
        c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = T.float32(0)


@T.prim_func
def wmma_fill_impl(c: T.handle) -> None:
    C = T.match_buffer(
        c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        T.reads([])
        T.writes(C[0:16, 0:16])
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
        T.evaluate(
            T.tvm_fill_fragment(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                T.float32(0),
                dtype="handle",
            )
        )


@T.prim_func
def wmma_store_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_store_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (16, 16), "float32", align=128, offset_factor=16, scope="global", strides=[s1, s0]
    )
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        threadIdx_x = T.env_thread("threadIdx.x")
        T.launch_thread(threadIdx_x, 32)
        T.evaluate(
            T.tvm_store_matrix_sync(
                A.data,
                16,
                16,
                16,
                A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                C.access_ptr("w"),
                s1,
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
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    I = T.dense_fixed(nb)
    J = T.sparse_variable(I, (mb, nnzb), (indptr, indices), "int32")
    J_detach = T.dense_fixed(mb)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), "float16")
    B = T.match_sparse_buffer(b, (J_detach, BJ, F), "float16")
    C = T.match_sparse_buffer(c, (I, BI, F), "float32")

    with T.iter([I, BI, F, J, BJ], "SSSRR", "bsrmm") as [
        vi,
        vbi,
        vf,
        vj,
        vbj,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + T.cast(A[vi, vj, vbi, vbj], "float32") * T.cast(
            B[vj, vbj, vf], "float32"
        )


# mod = tvm.IRModule.from_expr(bsrmm)
# mod = tvm.tir.transform.LowerSparseTIR()(mod)

# # print(mod["main"].script())

# bsr_mm = mod["main"]

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


@T.prim_func
def bsr_mm(
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
    A_data = T.match_buffer(a, [nnzb, blk, blk], dtype="float16")
    B_data = T.match_buffer(b, [mb, blk, feat_size], dtype="float16")
    C_data = T.match_buffer(c, [nb, blk, feat_size], dtype="float32")
    J_indptr = T.match_buffer(indptr, [nb + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnzb], dtype="int32")

    for v_vi, v_vbi, v_vf in T.grid(nb, blk, feat_size):
        with T.block("bsrmm0"):
            vi, vbi, vf = T.axis.remap("SSS", [v_vi, v_vbi, v_vf])
            T.reads(
                J_indptr[vi : vi + 2],
                J_indices[J_indptr[vi] + 0 : J_indptr[vi + 1] + 0],
                A_data[J_indptr[vi] + 0 : J_indptr[vi + 1] + 0, vbi, 0:blk],
                B_data[J_indices[J_indptr[vi]] + 0 : J_indices[J_indptr[vi + 1]] + 0, 0:blk, vf],
                C_data[vi, vbi, vf],
            )
            T.writes(C_data[vi, vbi, vf])
            T.block_attr({"sparse": True})
            with T.init():
                C_data[vi, vbi, vf] = T.float32(0)
            for v_vj, v_vbj in T.grid(J_indptr[vi + 1] - J_indptr[vi], blk):
                with T.block("bsrmm1"):
                    vj, vbj = T.axis.remap("RR", [v_vj, v_vbj])
                    T.reads(
                        J_indptr[vi],
                        J_indices[J_indptr[vi] + vj],
                        A_data[J_indptr[vi] + vj, vbi, vbj],
                        B_data[J_indices[J_indptr[vi] + vj], vbj, vf],
                        C_data[vi, vbi, vf],
                    )
                    T.writes(C_data[vi, vbi, vf])
                    T.block_attr({"sparse": True})
                    C_data[vi, vbi, vf] = C_data[vi, vbi, vf] + T.cast(
                        A_data[J_indptr[vi] + vj, vbi, vbj], "float32"
                    ) * T.cast(B_data[J_indices[J_indptr[vi] + vj], vbj, vf], "float32")


v_nb, v_mb, v_nnzb, v_blk, v_feat_size = bsr_mm.params[-5:]
bsr_mm = bsr_mm.specialize(
    {v_nb: nb, v_mb: mb, v_nnzb: nnzb, v_blk: block_size, v_feat_size: feat_size}
)

print(bsr_mm.script())


@T.prim_func
def specialized(
    A_data: T.Buffer[(51, 16, 16), "float16"],
    B_data: T.Buffer[(32, 16, 256), "float16"],
    C_data: T.Buffer[(32, 16, 256), "float32"],
    J_indptr: T.Buffer[(33,), "int32"],
    J_indices: T.Buffer[(51,), "int32"],
) -> None:
    for v_vi, v_vbi, v_vf in T.grid(32, 16, 256):
        with T.block("bsrmm0"):
            vi, vbi, vf = T.axis.remap("SSS", [v_vi, v_vbi, v_vf])
            T.reads(
                J_indptr[vi : vi + 2],
                J_indices[J_indptr[vi] : J_indptr[vi] + (J_indptr[vi + 1] - J_indptr[vi])],
                A_data[J_indptr[vi] : J_indptr[vi] + (J_indptr[vi + 1] - J_indptr[vi]), vbi, 0:16],
                B_data[
                    J_indices[J_indptr[vi]] : J_indices[J_indptr[vi]]
                    + (J_indices[J_indptr[vi + 1]] - J_indices[J_indptr[vi]]),
                    0:16,
                    vf,
                ],
                C_data[vi, vbi, vf],
            )
            T.writes(C_data[vi, vbi, vf])
            T.block_attr({"sparse": True})
            with T.init():
                C_data[vi, vbi, vf] = T.float32(0)
            for v_vj, v_vbj in T.grid(J_indptr[vi + 1] - J_indptr[vi], 16):
                with T.block("bsrmm1"):
                    vj, vbj = T.axis.remap("RR", [v_vj, v_vbj])
                    T.reads(
                        J_indptr[vi],
                        J_indices[J_indptr[vi] + vj],
                        A_data[J_indptr[vi] + vj, vbi, vbj],
                        B_data[J_indices[J_indptr[vi] + vj], vbj, vf],
                        C_data[vi, vbi, vf],
                    )
                    T.writes(C_data[vi, vbi, vf])
                    T.block_attr({"sparse": True})
                    C_data[vi, vbi, vf] = C_data[vi, vbi, vf] + T.cast(
                        A_data[J_indptr[vi] + vj, vbi, vbj], "float32"
                    ) * T.cast(B_data[J_indices[J_indptr[vi] + vj], vbj, vf], "float32")


sch = tir.Schedule(bsr_mm, debug_mask="all")
bsrmm0 = sch.get_block("bsrmm0")
A_local = sch.cache_read(bsrmm0, 2, "local")
B_local = sch.cache_read(bsrmm0, 3, "local")
C_local = sch.cache_write(bsrmm0, 0, "local")
i, bi, j = sch.get_loops(bsrmm0)
j, bj = sch.split(j, [None, 16])
# print(sch.mod["main"].script())


# for t in tqdm(range(0, 200)):
#     f = tvm.build(sch.mod["main"], target="llvm")

#     ctx = tvm.cpu(0)
#     A_indptr = tvm.nd.array(np.copy(indptr).astype("int32"), device=ctx)
#     A_indices = tvm.nd.array(np.copy(indices).astype("int32"), device=ctx)
#     A_data = tvm.nd.array(np.copy(data).astype("float16"), device=ctx)
#     X_nd = tvm.nd.array(np.copy(x.reshape(mb, block_size, feat_size)).astype("float16"), device=ctx)
#     Y_nd = tvm.nd.array(np.zeros((nb, block_size, feat_size), dtype="float32"), device=ctx)
#     f(A_data, X_nd, Y_nd, A_indptr, A_indices)
#     tvm.testing.assert_allclose(np.copy(y_ground_truth).reshape(nb, block_size, feat_size), Y_nd.numpy(), rtol=1e-5, atol=1e-5)


@T.prim_func
def split_reorder(
    A_data: T.Buffer[(51, 16, 16), "float16"],
    B_data: T.Buffer[(32, 16, 256), "float16"],
    C_data: T.Buffer[(32, 16, 256), "float32"],
    J_indptr: T.Buffer[(33,), "int32"],
    J_indices: T.Buffer[(51,), "int32"],
) -> None:
    A_data_local = T.alloc_buffer([51, 16, 16], dtype="float16", scope="local")
    B_data_local = T.alloc_buffer([32, 16, 256], dtype="float16", scope="local")
    C_data_local = T.alloc_buffer([32, 16, 256], dtype="float32", scope="local")
    for ax0, ax1, ax2 in T.grid(32, 16, 256):
        with T.block("B_data_local"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(B_data[v0, v1, v2])
            T.writes(B_data_local[v0, v1, v2])
            B_data_local[v0, v1, v2] = B_data[v0, v1, v2]
    for ax0, ax1, ax2 in T.grid(51, 16, 16):
        with T.block("A_data_local"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A_data[v0, v1, v2])
            T.writes(A_data_local[v0, v1, v2])
            A_data_local[v0, v1, v2] = A_data[v0, v1, v2]

    for v_vi, v_vf in T.grid(32, 16):
        with T.block("bsrmm0"):
            vi, vf = T.axis.remap("SS", [v_vi, v_vf])
            T.reads(
                J_indptr[vi : vi + 2],
                J_indices[J_indptr[vi] : J_indptr[vi + 1]],
                A_data_local[J_indptr[vi] : J_indptr[vi + 1], 0:16, 0:16],
                B_data_local[
                    J_indices[J_indptr[vi]] : J_indices[J_indptr[vi + 1]],
                    0:16,
                    vf * 16 : vf * 16 + 16,
                ],
                C_data_local[vi, 0:16, vf * 16 : vf * 16 + 16],
            )
            T.writes(C_data_local[vi, 0:16, vf * 16 : vf * 16 + 16])
            T.block_attr({"sparse": True})
            with T.init():
                for v_vbi_init, v_vbf_init in T.grid(16, 16):
                    with T.block("bsrmm_init"):
                        vbi_init, vbf_init = T.axis.remap("SS", [v_vbi_init, v_vbf_init])
                        T.block_attr({"sparse": True})
                        C_data_local[vi, vbi_init, vf * 16 + vbf_init] = T.float32(0)
            for v_vj in T.serial(0, J_indptr[vi + 1] - J_indptr[vi]):
                for v_vbi, v_vbf, v_vbj in T.grid(16, 16, 16):
                    with T.block("bsrmm1"):
                        vj, vbi, vbf, vbj = T.axis.remap("RSSR", [v_vj, v_vbi, v_vbf, v_vbj])
                        T.reads(
                            J_indptr[vi],
                            J_indices[J_indptr[vi] + vj],
                            A_data_local[J_indptr[vi] + vj, vbi, vbj],
                            B_data_local[J_indices[J_indptr[vi] + vj], vbj, vf * 16 + vbf],
                            C_data_local[vi, vbi, vf * 16 + vbf],
                        )
                        T.writes(C_data_local[vi, vbi, vf * 16 + vbf])
                        T.block_attr({"sparse": True})
                        C_data_local[vi, vbi, vf * 16 + vbf] = C_data_local[
                            vi, vbi, vf * 16 + vbf
                        ] + T.cast(A_data_local[J_indptr[vi] + vj, vbi, vbj], "float32") * T.cast(
                            B_data_local[J_indices[J_indptr[vi] + vj], vbj, vf * 16 + vbf],
                            "float32",
                        )

    for ax0, ax1, ax2 in T.grid(32, 16, 256):
        with T.block("C_data_local"):
            v0, v1, v2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(C_data_local[v0, v1, v2])
            T.writes(C_data[v0, v1, v2])
            C_data[v0, v1, v2] = C_data_local[v0, v1, v2]


# for t in tqdm(range(0, 200)):
#     f = tvm.build(split_reorder, target="llvm")

#     ctx = tvm.cpu(0)
#     A_indptr = tvm.nd.array(np.copy(indptr).astype("int32"), device=ctx)
#     A_indices = tvm.nd.array(np.copy(indices).astype("int32"), device=ctx)
#     A_data = tvm.nd.array(np.copy(data).astype("float16"), device=ctx)
#     X_nd = tvm.nd.array(np.copy(x.reshape(mb, block_size, feat_size)).astype("float16"), device=ctx)
#     Y_nd = tvm.nd.array(np.zeros((nb, block_size, feat_size), dtype="float32"), device=ctx)
#     f(A_data, X_nd, Y_nd, A_indptr, A_indices)
#     tvm.testing.assert_allclose(np.copy(y_ground_truth).reshape(nb, block_size, feat_size), Y_nd.numpy(), rtol=1e-5, atol=1e-5)


sch = tir.Schedule(split_reorder, debug_mask="all")

C_data_local = sch.get_block("C_data_local")
bsrmm0 = sch.get_block("bsrmm0")
vi, vf = sch.get_loops(bsrmm0)
sch.reverse_compute_at(C_data_local, vf)
# print(sch.mod["main"].script())


@T.prim_func
def compute_at(
    A_data: T.Buffer[(51, 16, 16), "float16"],
    B_data: T.Buffer[(32, 16, 256), "float16"],
    C_data: T.Buffer[(32, 16, 256), "float32"],
    J_indptr: T.Buffer[(33,), "int32"],
    J_indices: T.Buffer[(51,), "int32"],
) -> None:
    A_data_local = T.alloc_buffer([51, 16, 16], dtype="float16", scope="local")
    B_data_local = T.alloc_buffer([32, 16, 256], dtype="float16", scope="local")
    C_data_local = T.alloc_buffer([32, 16, 256], dtype="float32", scope="local")
    for v_vi in T.serial(0, 32):
        for v_vf in T.serial(0, 16):
            with T.block("bsrmm0"):
                vi, vf = T.axis.remap("SS", [v_vi, v_vf])
                T.reads(
                    J_indptr[vi : vi + 2],
                    J_indices[J_indptr[vi] : J_indptr[vi + 1]],
                    A_data[J_indptr[vi] : J_indptr[vi + 1], 0:16, 0:16],
                    B_data[
                        J_indices[J_indptr[vi]] : J_indices[J_indptr[vi + 1]],
                        0:16,
                        vf * 16 : vf * 16 + 16,
                    ],
                    C_data_local[vi, 0:16, vf * 16 : vf * 16 + 16],
                )
                T.writes(
                    A_data_local[J_indptr[vi] : J_indptr[vi + 1], 0:16, 0:16],
                    B_data_local[
                        J_indices[J_indptr[vi]] : J_indices[J_indptr[vi + 1]],
                        0:16,
                        vf * 16 : vf * 16 + 16,
                    ],
                    C_data_local[vi, 0:16, vf * 16 : vf * 16 + 16],
                )
                T.block_attr({"sparse": True})
                with T.init():
                    for v_vbi_init, v_vbf_init in T.grid(16, 16):
                        with T.block("bsrmm_init"):
                            vbi_init, vbf_init = T.axis.remap("SS", [v_vbi_init, v_vbf_init])
                            T.reads()
                            T.writes(C_data_local[vi, vbi_init, vf * 16 + vbf_init])
                            T.block_attr({"sparse": True})
                            C_data_local[vi, vbi_init, vf * 16 + vbf_init] = T.float32(0)
                for v_vj in T.serial(0, J_indptr[vi + 1] - J_indptr[vi]):
                    for ax1, ax2 in T.grid(16, 16):
                        with T.block("B_data_local"):
                            v0 = T.axis.S(32, J_indices[J_indptr[vi] + v_vj])
                            v1 = T.axis.S(16, ax1)
                            v2 = T.axis.S(256, vf * 16 + ax2)
                            T.reads(B_data[v0, v1, v2])
                            T.writes(B_data_local[v0, v1, v2])
                            B_data_local[v0, v1, v2] = B_data[v0, v1, v2]
                    for ax1, ax2 in T.grid(16, 16):
                        with T.block("A_data_local"):
                            v0 = T.axis.S(51, J_indptr[vi] + v_vj)
                            v1, v2 = T.axis.remap("SS", [ax1, ax2])
                            T.reads(A_data[v0, v1, v2])
                            T.writes(A_data_local[v0, v1, v2])
                            A_data_local[v0, v1, v2] = A_data[v0, v1, v2]
                    for v_vbi, v_vbf, v_vbj in T.grid(16, 16, 16):
                        with T.block("bsrmm1"):
                            vj, vbi, vbf, vbj = T.axis.remap("RSSR", [v_vj, v_vbi, v_vbf, v_vbj])
                            T.reads(
                                J_indptr[vi],
                                J_indices[J_indptr[vi] + vj],
                                A_data_local[J_indptr[vi] + vj, vbi, vbj],
                                B_data_local[J_indices[J_indptr[vi] + vj], vbj, vf * 16 + vbf],
                                C_data_local[vi, vbi, vf * 16 + vbf],
                            )
                            T.writes(C_data_local[vi, vbi, vf * 16 + vbf])
                            T.block_attr({"sparse": True})
                            C_data_local[vi, vbi, vf * 16 + vbf] = C_data_local[
                                vi, vbi, vf * 16 + vbf
                            ] + T.cast(
                                A_data_local[J_indptr[vi] + vj, vbi, vbj], "float32"
                            ) * T.cast(
                                B_data_local[J_indices[J_indptr[vi] + vj], vbj, vf * 16 + vbf],
                                "float32",
                            )
            for ax0, ax1 in T.grid(16, 16):
                with T.block("C_data_local"):
                    v0, v1 = T.axis.remap("SS", [v_vi, ax0])
                    v2 = T.axis.spatial(256, v_vf * 16 + ax1)
                    T.reads(C_data_local[v0, v1, v2])
                    T.writes(C_data[v0, v1, v2])
                    C_data[v0, v1, v2] = C_data_local[v0, v1, v2]


# for t in tqdm(range(0, 100)):
#     f = tvm.build(compute_at, target="llvm")
#     ctx = tvm.cpu(0)
#     A_indptr = tvm.nd.array(np.copy(indptr).astype("int32"), device=ctx)
#     A_indices = tvm.nd.array(np.copy(indices).astype("int32"), device=ctx)
#     A_data = tvm.nd.array(np.copy(data).astype("float16"), device=ctx)
#     X_nd = tvm.nd.array(np.copy(x.reshape(mb, block_size, feat_size)).astype("float16"), device=ctx)
#     Y_nd = tvm.nd.array(np.zeros((nb, block_size, feat_size), dtype="float32"), device=ctx)
#     f(A_data, X_nd, Y_nd, A_indptr, A_indices)
#     tvm.testing.assert_allclose(np.copy(y_ground_truth).reshape(nb, block_size, feat_size), Y_nd.numpy(), rtol=1e-5, atol=1e-5)
# evaluator = f.time_evaluator(f.entry_name, ctx, number=10)
# print(evaluator(A_data, X_nd, Y_nd, A_indptr, A_indices))


@T.prim_func
def bind_block_axis(
    A_data: T.Buffer[(51, 16, 16), "float16"],
    B_data: T.Buffer[(32, 16, 256), "float16"],
    C_data: T.Buffer[(32, 16, 256), "float32"],
    J_indptr: T.Buffer[(33,), "int32"],
    J_indices: T.Buffer[(51,), "int32"],
) -> None:
    A_data_local = T.alloc_buffer([51, 16, 16], dtype="float16", scope="local")
    B_data_local = T.alloc_buffer([32, 16, 256], dtype="float16", scope="local")
    C_data_local = T.alloc_buffer([32, 16, 256], dtype="float32", scope="local")
    for v_vi in T.thread_binding(0, 32, thread="blockIdx.x"):
        for v_vf in T.thread_binding(0, 16, thread="blockIdx.y"):
            with T.block("bsrmm0"):
                vi, vf = T.axis.remap("SS", [v_vi, v_vf])
                T.reads(
                    J_indptr[vi : vi + 2],
                    J_indices[J_indptr[vi] : J_indptr[vi + 1]],
                    A_data[J_indptr[vi] : J_indptr[vi + 1], 0:16, 0:16],
                    B_data[
                        J_indices[J_indptr[vi]] : J_indices[J_indptr[vi + 1]],
                        0:16,
                        vf * 16 : vf * 16 + 16,
                    ],
                    C_data_local[vi, 0:16, vf * 16 : vf * 16 + 16],
                )
                T.writes(
                    A_data_local[J_indptr[vi] : J_indptr[vi + 1], 0:16, 0:16],
                    B_data_local[
                        J_indices[J_indptr[vi]] : J_indices[J_indptr[vi + 1]],
                        0:16,
                        vf * 16 : vf * 16 + 16,
                    ],
                    C_data_local[vi, 0:16, vf * 16 : vf * 16 + 16],
                )
                T.block_attr({"sparse": True})
                with T.init():
                    for v_vbi_init, v_vbf_init in T.grid(16, 16):
                        with T.block("bsrmm_init"):
                            vbi_init, vbf_init = T.axis.remap("SS", [v_vbi_init, v_vbf_init])
                            T.reads()
                            T.writes(C_data_local[vi, vbi_init, vf * 16 + vbf_init])
                            T.block_attr({"sparse": True})
                            C_data_local[vi, vbi_init, vf * 16 + vbf_init] = T.float32(0)
                for v_vj in T.serial(0, J_indptr[vi + 1] - J_indptr[vi]):
                    for ax1, ax2 in T.grid(16, 16):
                        with T.block("B_data_local"):
                            v0 = T.axis.S(32, J_indices[J_indptr[vi] + v_vj])
                            v1 = T.axis.S(16, ax1)
                            v2 = T.axis.S(256, vf * 16 + ax2)
                            T.reads(B_data[v0, v1, v2])
                            T.writes(B_data_local[v0, v1, v2])
                            B_data_local[v0, v1, v2] = B_data[v0, v1, v2]
                    for ax1, ax2 in T.grid(16, 16):
                        with T.block("A_data_local"):
                            v0 = T.axis.S(51, J_indptr[vi] + v_vj)
                            v1, v2 = T.axis.remap("SS", [ax1, ax2])
                            T.reads(A_data[v0, v1, v2])
                            T.writes(A_data_local[v0, v1, v2])
                            A_data_local[v0, v1, v2] = A_data[v0, v1, v2]
                    for v_vbi, v_vbf, v_vbj in T.grid(16, 16, 16):
                        with T.block("bsrmm1"):
                            vj, vbi, vbf, vbj = T.axis.remap("RSSR", [v_vj, v_vbi, v_vbf, v_vbj])
                            T.reads(
                                J_indptr[vi],
                                J_indices[J_indptr[vi] + vj],
                                A_data_local[J_indptr[vi] + vj, vbi, vbj],
                                B_data_local[J_indices[J_indptr[vi] + vj], vbj, vf * 16 + vbf],
                                C_data_local[vi, vbi, vf * 16 + vbf],
                            )
                            T.writes(C_data_local[vi, vbi, vf * 16 + vbf])
                            T.block_attr({"sparse": True})
                            C_data_local[vi, vbi, vf * 16 + vbf] = C_data_local[
                                vi, vbi, vf * 16 + vbf
                            ] + T.cast(
                                A_data_local[J_indptr[vi] + vj, vbi, vbj], "float32"
                            ) * T.cast(
                                B_data_local[J_indices[J_indptr[vi] + vj], vbj, vf * 16 + vbf],
                                "float32",
                            )
            for ax0, ax1 in T.grid(16, 16):
                with T.block("C_data_local"):
                    v0, v1 = T.axis.remap("SS", [v_vi, ax0])
                    v2 = T.axis.spatial(256, v_vf * 16 + ax1)
                    T.reads(C_data_local[v0, v1, v2])
                    T.writes(C_data[v0, v1, v2])
                    C_data[v0, v1, v2] = C_data_local[v0, v1, v2]


# for t in tqdm(range(0, 100)):
#     f = tvm.build(bind_block_axis, target="cuda")
#     ctx = tvm.cuda(0)
#     A_indptr = tvm.nd.array(np.copy(indptr).astype("int32"), device=ctx)
#     A_indices = tvm.nd.array(np.copy(indices).astype("int32"), device=ctx)
#     A_data = tvm.nd.array(np.copy(data).astype("float16"), device=ctx)
#     X_nd = tvm.nd.array(np.copy(x.reshape(mb, block_size, feat_size)).astype("float16"), device=ctx)
#     Y_nd = tvm.nd.array(np.zeros((nb, block_size, feat_size), dtype="float32"), device=ctx)
#     f(A_data, X_nd, Y_nd, A_indptr, A_indices)
#     tvm.testing.assert_allclose(np.copy(y_ground_truth).reshape(nb, block_size, feat_size), Y_nd.numpy(), rtol=1e-5, atol=1e-5)
# evaluator = f.time_evaluator(f.entry_name, ctx, number=10)
# print(evaluator(A_data, X_nd, Y_nd, A_indptr, A_indices))


@T.prim_func
def blockized_B_data_copy(
    A_data: T.Buffer[(51, 16, 16), "float16"],
    B_data: T.Buffer[(32, 16, 256), "float16"],
    C_data: T.Buffer[(32, 16, 256), "float32"],
    J_indptr: T.Buffer[(33,), "int32"],
    J_indices: T.Buffer[(51,), "int32"],
) -> None:
    A_data_local = T.alloc_buffer([51, 16, 16], dtype="float16", scope="wmma.matrix_a")
    B_data_local = T.alloc_buffer([32, 16, 256], dtype="float16", scope="wmma.matrix_b")
    C_data_local = T.alloc_buffer([32, 16, 256], dtype="float32", scope="wmma.accumulator")
    for v_vi in T.thread_binding(0, 32, thread="blockIdx.x"):
        for v_vf in T.thread_binding(0, 16, thread="blockIdx.y"):
            with T.block("bsrmm0"):
                vi, vf = T.axis.remap("SS", [v_vi, v_vf])
                T.reads(
                    J_indptr[vi : vi + 2],
                    J_indices[J_indptr[vi] : J_indptr[vi + 1]],
                    A_data[J_indptr[vi] : J_indptr[vi + 1], 0:16, 0:16],
                    B_data[
                        J_indices[J_indptr[vi]] : J_indices[J_indptr[vi + 1]],
                        0:16,
                        vf * 16 : vf * 16 + 16,
                    ],
                    C_data_local[vi, 0:16, vf * 16 : vf * 16 + 16],
                )
                T.writes(
                    A_data_local[J_indptr[vi] : J_indptr[vi + 1], 0:16, 0:16],
                    B_data_local[
                        J_indices[J_indptr[vi]] : J_indices[J_indptr[vi + 1]],
                        0:16,
                        vf * 16 : vf * 16 + 16,
                    ],
                    C_data_local[vi, 0:16, vf * 16 : vf * 16 + 16],
                )
                T.block_attr({"sparse": True})
                for v_vbi_init, v_vbf_init in T.grid(16, 16):
                    with T.block("bsrmm_init"):
                        vbi_init, vbf_init = T.axis.remap("SS", [v_vbi_init, v_vbf_init])
                        T.reads()
                        T.writes(C_data_local[vi, vbi_init, vf * 16 + vbf_init])
                        C_data_local[vi, vbi_init, vf * 16 + vbf_init] = T.float32(0)
                for v_vj in T.serial(0, J_indptr[vi + 1] - J_indptr[vi]):
                    with T.block("B_data_local_o"):
                        v0 = T.axis.S(32, J_indices[J_indptr[vi] + v_vj])
                        v1_o = T.axis.S(1, 0)
                        v2_o = T.axis.S(16, vf)
                        T.reads(B_data[v0, 0:16, v2_o * 16 : v2_o * 16 + 16])
                        T.writes(B_data_local[v0, 0:16, v2_o * 16 : v2_o * 16 + 16])
                        for ax1, ax2 in T.grid(16, 16):
                            with T.block("B_data_local"):
                                v1 = T.axis.S(16, ax1)
                                v2 = T.axis.S(16, ax2)
                                T.reads(B_data[v0, v1, v2_o * 16 + v2])
                                T.writes(B_data_local[v0, v1, v2_o * 16 + v2])
                                B_data_local[v0, v1, v2_o * 16 + v2] = B_data[
                                    v0, v1, v2_o * 16 + v2
                                ]
                    for ax1, ax2 in T.grid(16, 16):
                        with T.block("A_data_local"):
                            v0 = T.axis.S(51, J_indptr[vi] + v_vj)
                            v1, v2 = T.axis.remap("SS", [ax1, ax2])
                            T.reads(A_data[v0, v1, v2])
                            T.writes(A_data_local[v0, v1, v2])
                            A_data_local[v0, v1, v2] = A_data[v0, v1, v2]
                    for v_vbi, v_vbf, v_vbj in T.grid(16, 16, 16):
                        with T.block("bsrmm1"):
                            vj, vbi, vbf, vbj = T.axis.remap("RSSR", [v_vj, v_vbi, v_vbf, v_vbj])
                            # T.reads(J_indptr[vi], J_indices[J_indptr[vi] + vj], C_data_local[vi, vbi, vf * 16 + vbf], A_data_local[J_indptr[vi] + vj, vbi, vbj], B_data_local[J_indices[J_indptr[vi] + vj], vbj, vf * 16 + vbf])
                            T.reads(
                                C_data_local[vi, vbi, vf * 16 + vbf],
                                A_data_local[J_indptr[vi] + vj, vbi, vbj],
                                B_data_local[J_indices[J_indptr[vi] + vj], vbj, vf * 16 + vbf],
                            )
                            T.writes(C_data_local[vi, vbi, vf * 16 + vbf])
                            T.block_attr({"sparse": True})
                            C_data_local[vi, vbi, vf * 16 + vbf] = C_data_local[
                                vi, vbi, vf * 16 + vbf
                            ] + T.cast(
                                A_data_local[J_indptr[vi] + vj, vbi, vbj], "float32"
                            ) * T.cast(
                                B_data_local[J_indices[J_indptr[vi] + vj], vbj, vf * 16 + vbf],
                                "float32",
                            )
            for ax0, ax1 in T.grid(16, 16):
                with T.block("C_data_local"):
                    v0, v1 = T.axis.remap("SS", [v_vi, ax0])
                    v2 = T.axis.spatial(256, v_vf * 16 + ax1)
                    T.reads(C_data_local[v0, v1, v2])
                    T.writes(C_data[v0, v1, v2])
                    C_data[v0, v1, v2] = C_data_local[v0, v1, v2]


sch = tir.Schedule(blockized_B_data_copy, debug_mask="all")

sch.tensorize(sch.get_loops(sch.get_block("bsrmm1"))[-3], "wmma_sync")
sch.tensorize(sch.get_block("B_data_local_o"), "wmma_load_b")
sch.tensorize(sch.get_loops(sch.get_block("A_data_local"))[-2], "wmma_load_a")
sch.tensorize(sch.get_loops(sch.get_block("bsrmm_init"))[-2], "wmma_fill")
sch.tensorize(sch.get_loops(sch.get_block("C_data_local"))[-2], "wmma_store")


print(sch.mod["main"].script())
print(tvm.build(sch.mod["main"], target="cuda").imported_modules[0].get_source())

for t in tqdm(range(0, 2)):
    f = tvm.build(bind_block_axis, target="cuda")
    ctx = tvm.cuda(0)
    A_indptr = tvm.nd.array(np.copy(indptr).astype("int32"), device=ctx)
    A_indices = tvm.nd.array(np.copy(indices).astype("int32"), device=ctx)
    A_data = tvm.nd.array(np.copy(data).astype("float16"), device=ctx)
    X_nd = tvm.nd.array(np.copy(x.reshape(mb, block_size, feat_size)).astype("float16"), device=ctx)
    Y_nd = tvm.nd.array(np.zeros((nb, block_size, feat_size), dtype="float32"), device=ctx)
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(
        np.copy(y_ground_truth).reshape(nb, block_size, feat_size),
        Y_nd.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
evaluator = f.time_evaluator(f.entry_name, ctx, number=10)
print("w/o Tensor Cores:")
print(evaluator(A_data, X_nd, Y_nd, A_indptr, A_indices))

for t in tqdm(range(0, 2)):
    f = tvm.build(sch.mod["main"], target="cuda")
    ctx = tvm.cuda(0)
    A_indptr = tvm.nd.array(np.copy(indptr).astype("int32"), device=ctx)
    A_indices = tvm.nd.array(np.copy(indices).astype("int32"), device=ctx)
    A_data = tvm.nd.array(np.copy(data).astype("float16"), device=ctx)
    X_nd = tvm.nd.array(np.copy(x.reshape(mb, block_size, feat_size)).astype("float16"), device=ctx)
    Y_nd = tvm.nd.array(np.zeros((nb, block_size, feat_size), dtype="float32"), device=ctx)
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)
    tvm.testing.assert_allclose(
        np.copy(y_ground_truth).reshape(nb, block_size, feat_size),
        Y_nd.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
print("with Tensor Cores:")
evaluator = f.time_evaluator(f.entry_name, ctx, number=10)
print(evaluator(A_data, X_nd, Y_nd, A_indptr, A_indices))
