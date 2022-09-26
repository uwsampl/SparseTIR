"""FusedMM
Related work: https://arxiv.org/pdf/2011.06391.pdf
"""

import tvm
import tvm.testing
import tvm.tir as tir
import argparse
from tvm.script import tir as T
from tvm.sparse import lower_sparse_buffer, lower_sparse_iter


"""
softmax(Q^T * K) * V

sparse matrix: [m, n] with #nonzeros nnz
"""


@T.prim_func
def fusedmm(
    q: T.handle,
    k: T.handle,
    v: T.handle,
    o: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
    feat_size: T.int32,
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "sparse_tir_level": 2})
    I = T.dense_fixed(m, idtype="int32")
    J = T.sparse_variable(I, (n, nnz), (indptr, indices), idtype="int32")
    F = T.dense_fixed(feat_size, idtype="int32")
    J_ = T.dense_fixed(n, idtype="int32")

    Q = T.match_sparse_buffer(q, [I, F], "float32")
    K = T.match_sparse_buffer(k, [J_, F], "float32")
    V = T.match_sparse_buffer(v, [J_, F], "float32")
    O = T.match_sparse_buffer(o, [I, F], "float32")

    score = T.alloc_sparse_buffer([I, J], "float32")
    temp = T.alloc_sparse_buffer([I,], "float32")
    temp1 = T.alloc_sparse_buffer([I,], "float32")
    softmax = T.alloc_sparse_buffer([I, J], "float32")
    # Q^T * K
    with T.iter([I, J, F], "SSR", "sddmm") as [i, j, f]:
        with T.init():
            score[i, j] = T.float32(0)
        score[i, j] += Q[i, f] * K[j, f]

    # softmax
    with T.iter([I], "S", "softmax") as [i]:
        with T.iter([J], "R", "computer_max") as [j]:
            with T.init():
                temp[i] = score[i, j]
            temp[i] = T.max(temp[i], score[i, j])
        with T.iter([J], "R", "sum_of_exp") as [j]:
            with T.init():
                temp1[i] = T.float32(0)
            temp1[i] += T.exp(score[i, j] - temp[i], dtype="float32")
        with T.iter([J], "S", "normalize") as [j]:
            softmax[i, j] = T.exp(score[i, j], dtype="float32") / temp1[i]
        
    # softmax * V
    with T.iter([I, J, F], "SRS", "spmm") as [i, j, f]:
        with T.init():
            O[i, f] = T.float32(0)
        O[i, f] = O[i, f] + softmax[i, j] * V[j, f]


def bench_fusedmm():
    mod = tvm.IRModule.from_expr(fusedmm)
    mod = lower_sparse_iter(mod)
    sch = tir.Schedule(mod)
    spmm_blk_outer = sch.get_block("spmm0")
    print(mod["main"].script())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FusedMM in Sparse-TIR")
    parser.add_argument("--dataset", "-d", type=str, default="arxiv", help="dataset name")
    args = parser.parse_args()
    name = args.dataset
    bench_fusedmm()
