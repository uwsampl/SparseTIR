import tvm
import tvm.testing
from tvm.runtime.ndarray import device
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
from tvm.script import tir as T


@T.prim_func
def butterfly(w1: T.handle, w2: T.handle, w3: T.handle, w4: T.handle, x: T.handle, y: T.handle) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    W1 = T.match_buffer(w1, (16, 2), "float32")
    W2 = T.match_buffer(w2, (16, 2), "float32")
    W3 = T.match_buffer(w3, (16, 2), "float32")
    W4 = T.match_buffer(w4, (16, 2), "float32")
    X = T.match_buffer(x, (16, 64), "float32")
    Y = T.match_buffer(y, (16, 64), "float32")

    for i, j, k in T.grid(16, 2, 64):
        with T.block("wx"):
            vi, vj, vk = T.axis.remap("SRS", [i, j, k])
            with T.init():
                Y[vi, vk] = 0.
            Y[vi, vk] = Y[vi, vk] +\
                W1[vi, vj] * X[vj * 8 + T.floormod(vi, 8), vk] +\
                W2[vi, vj] * X[T.floordiv(vi, 8) * 8 + vj * 4 + T.floormod(vi, 4), vk] +\
                W3[vi, vj] * X[T.floordiv(vi, 4) * 4 + vj * 2 + T.floormod(vi, 2), vk] +\
                W4[vi, vj] * X[T.floordiv(vi, 2) * 2 + vj, vk]


def test_butterfly():
    sch = tir.Schedule(butterfly)
    print(sch.mod["main"].script())


if __name__ == "__main__":
    test_butterfly()
