from tvm.script import tir as T
from sparse_tir_scripts import rgcn_hetero_forward
import tvm


def test_schedule_rgcn():
    func = rgcn_hetero_forward
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.LowerSparseIter()(mod)
    sch = tvm.tir.Schedule(mod)

    blk0 = sch.get_block("rgcn-hetero-forward0")
    blk1 = sch.get_block("rgcn-hetero-forward1")
    blk2 = sch.get_block("rgcn-hetero-forward2")
    read_blk = sch.cache_read(blk1, 2, "shared")
    write_blk = sch.cache_write(blk2, 0, "local")
    f_out, r = sch.get_loops(blk0)
    (i,) = sch.get_loops(blk1)
    j, f_in = sch.get_loops(blk2)
    sch.bind(f_in, "threadIdx.x")
    sch.reorder(f_in, j)
    sch.decompose_reduction(blk2, f_in)
    i1, i2 = sch.split(i, [None, 8])
    sch.bind(i2, "blockIdx.x")
    sch.bind(r, "blockIdx.y")
    sch.bind(f_out, "threadIdx.y")
    _, _, ax2 = sch.get_loops(read_blk)
    sch.bind(ax2, "threadIdx.x")
    mod = tvm.tir.transform.LowerSparseBuffer()(sch.mod)
    print(mod["main"].script())


if __name__ == "__main__":
    test_schedule_rgcn()
