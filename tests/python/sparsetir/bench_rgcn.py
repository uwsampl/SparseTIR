from dgl.heterograph import DGLHeteroGraph
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
from ogb.linkproppred import DglLinkPropPredDataset
from sparse_tir_scripts import rgcn_forward


def get_dataset_by_name(name: str):
    if name == "aifb":
        return AIFBDataset()
    elif name == "mutag":
        return MUTAGDataset()
    elif name == "bgs":
        return BGSDataset()
    elif name == "am":
        return AMDataset()
    elif name == 'biokg':
        return DglLinkPropPredDataset(name='ogbl-biokg')
    else:
        raise KeyError("Unknown dataset {}.".format(name))


class TorchOpTimer(object):
    def __enter__(self):
        self.start_event = th.cuda.Event(enable_timing=True)
        self.end_event = th.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end_event.record()
        th.cuda.synchronize()  # Wait for the events to be recorded!
        self.time = self.start_event.elapsed_time(self.end_event)


def prepare_hetero_graph_simplified(g: dgl.DGLHeteroGraph):
    ntype_pointer = np.cumsum([0] + [g.number_of_nodes(ntype) for ntype in g.ntypes])

    etype_pointer = [0]
    for etype in g.canonical_etypes:
        g_sub = g[etype]
        etype_pointer.append(etype_pointer[-1] + g_sub.num_edges())

    return {
        "ntype_node_pointer": th.IntTensor(ntype_pointer).cuda(),
        "etype_edge_pointer": th.IntTensor(etype_pointer).cuda(),
    }


def test_rgcn(g: DGLHeteroGraph, feat_size: int):
    g = g.to(0)
    feat = th.rand(g.num_src_nodes(), feat_size).to(0) / 100
    out = th.zeros(g.num_dst_nodes(), feat_size).to(0) / 100
    weight = th.rand(g.num_rels, feat_size, feat_size).to(0)
    indptr, indices, eid = g.adj_sparse(fmt="csc")
    etype = g.edata[dgl.ETYPE][eid]

    cold_start = 3
    total = 10
    accum = 0

    # dgl-lowmem
    try:
        g.srcdata["feat"] = feat.unsqueeze(-1)
        us, vs = g.edges()
        feat_transformed = feat[us]
        msg = th.zeros(g.num_edges(), feat_size).to(0)
        weight_T = weight.permute(0, 2, 1).contiguous()
        for epoch in range(10):
            with TorchOpTimer() as timer:
                with th.no_grad():
                    for i in range(1, len(g.etype_pointer)):
                        start = g.etype_pointer[i - 1]
                        end = g.etype_pointer[i]
                        msg[start:end] = feat_transformed[start:end] @ weight_T[i - 1]
                    y_dgl_lowmem = dgl.ops.copy_e_sum(g, msg)
            if epoch >= cold_start:
                accum += timer.time
        print("dgl-lowmem:\t\t {}".format(accum / (total - cold_start)))
    except RuntimeError as err:
        print("dgl-lowmem: OOM")
        y_dgl_lowmem = None
    except BaseException as err:
        print(err)
        raise

    # dgl-bmm

    def msg_func(edges):
        h = edges.src["feat"]
        W = weight[edges.data[dgl.ETYPE]]
        return {"msg": W @ h}

    try:
        g.srcdata["feat"] = feat.unsqueeze(-1)
        for epoch in range(10):
            with TorchOpTimer() as timer:
                with th.no_grad():
                    g.update_all(msg_func, fn.sum("msg", "y"))
                    y_dgl = g.dstdata["y"].squeeze(-1)
            if epoch >= cold_start:
                accum += timer.time
        print("dgl-bmm:\t\t {}".format(accum / (total - cold_start)))
    except RuntimeError as err:
        print("dgl-bmm: OOM")
        y_dgl = None
    except BaseException as err:
        print(err)
        raise

    # tir
    N, R, FEAT_SIZE, NNZ = rgcn_forward.params[-4:]
    mod = tvm.IRModule.from_expr(
        rgcn_forward.specialize(
            {N: g.number_of_nodes(), R: g.num_rels, FEAT_SIZE: feat_size, NNZ: g.number_of_edges()}
        )
    )
    mod = tvm.tir.transform.LowerSparseIter()(mod)
    sch = tir.Schedule(mod["main"])

    outer = sch.get_block("rgcn-forward0")
    inner = sch.get_block("rgcn-forward1")
    i, f_out = sch.get_loops(outer)
    j, f_in = sch.get_loops(inner)
    sch.bind(i, "blockIdx.x")
    sch.bind(f_out, "threadIdx.y")
    sch.bind(f_in, "threadIdx.x")
    mod = tvm.tir.transform.LowerSparseBuffer()(sch.mod)
    f = tvm.build(mod, target="cuda")

    E = tvm.nd.array(etype.cpu().numpy().astype("int32"), device=tvm.cuda(0))
    W = tvm.nd.array(weight.view(-1).cpu().numpy().astype("float32"), device=tvm.cuda(0))
    X = tvm.nd.array(feat.view(-1).cpu().numpy().astype("float32"), device=tvm.cuda(0))
    Y = tvm.nd.array(out.view(-1).cpu().numpy().astype("float32"), device=tvm.cuda(0))
    indptr = tvm.nd.array(indptr.cpu().numpy().astype("int32"), device=tvm.cuda(0))
    indices = tvm.nd.array(indices.cpu().numpy().astype("int32"), device=tvm.cuda(0))

    cold_start = 3
    total = 10
    accum = 0

    for epoch in range(10):
        with TorchOpTimer() as timer:
            f(E, W, X, Y, indptr, indices)
        if epoch >= cold_start:
            accum += timer.time

    print("sparse-tir:\t\t {}".format(accum / (total - cold_start)))

    if y_dgl is not None:
        tvm.testing.assert_allclose(y_dgl.view(-1).cpu().numpy(), Y.numpy(), rtol=1e-2)
    if y_dgl_lowmem is not None:
        tvm.testing.assert_allclose(y_dgl_lowmem.view(-1).cpu().numpy(), Y.numpy(), rtol=1e-2)


if __name__ == "__main__":
    for feat_size in [4, 8, 16, 32]:
        for name in ["biokg"]:#["aifb", "mutag", "bgs", "am"]:
            print("dataset {}, feat_size={}:".format(name, feat_size))
            dataset = get_dataset_by_name(name)
            g = dataset[0]
            type_pointers = prepare_hetero_graph_simplified(g)
            g = dgl.to_homogeneous(g)
            g.ntype_pointer = type_pointers["ntype_node_pointer"]
            g.etype_pointer = type_pointers["etype_edge_pointer"]
            g.num_ntypes = max(g.ndata[dgl.NTYPE]).item() + 1
            g.num_rels = max(g.edata[dgl.ETYPE]).item() + 1
            test_rgcn(g, feat_size)
