import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
import dgl
import dgl.function as fn
import torch as th
from tvm.script import tir as T
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from sparse_tir_scripts import rgcn_hetero_forward_2
from tvm.sparse import lower_sparse_iter, lower_sparse_buffer


def get_dataset_by_name(name: str):
    if name == "aifb":
        return AIFBDataset()
    elif name == "mutag":
        return MUTAGDataset()
    elif name == "bgs":
        return BGSDataset()
    elif name == "am":
        return AMDataset()
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
        "ntype_node_pointer": th.IntTensor(ntype_pointer),
        "etype_edge_pointer": th.IntTensor(etype_pointer),
    }


def get_ground_truth(g_homo: dgl.DGLHeteroGraph, feat: th.Tensor, weight: th.Tensor) -> th.Tensor:
    feat_size = feat.shape[-1]
    g_homo = g_homo.to(0)
    weight_T = weight.permute(0, 2, 1).contiguous()
    try:
        g_homo.srcdata["feat"] = feat.unsqueeze(-1)
        us, vs = g_homo.edges()
        feat_transformed = feat[us]
        msg = th.zeros(g_homo.num_edges(), feat_size).to(0)
        with th.no_grad():
            for i in range(1, len(g_homo.etype_pointer)):
                start = g_homo.etype_pointer[i - 1]
                end = g_homo.etype_pointer[i]
                msg[start:end] = feat_transformed[start:end] @ weight_T[i - 1]
            y_dgl_lowmem = dgl.ops.copy_e_sum(g_homo, msg)
    except RuntimeError as err:
        print("dgl-lowmem: OOM")
        y_dgl_lowmem = None
    return y_dgl_lowmem


def test_lower_rgcn_hetero(
    g: dgl.DGLHeteroGraph,
    feat_size: int,
    feat,
    weight,
    ground_truth_y,
    split_factor_f: int,
    bucket_size: int,
):
    N, R, GROUP, FEAT_SIZE, NNZ_I, NNZ_J = rgcn_hetero_forward_2.params[-6:]
    n = g.num_nodes()
    r = len(g.etypes)
    nnz_j = g.num_edges()

    out = np.zeros((n * feat_size))
    W = tvm.nd.array(weight.astype("float32"), device=tvm.cuda(0))
    X = tvm.nd.array(feat.astype("float32"), device=tvm.cuda(0))
    Y = tvm.nd.array(out.astype("float32"), device=tvm.cuda(0))

    indptr_i = [th.LongTensor([0])]
    indices_i = []
    indptr_j = [th.LongTensor([0])]
    indices_j = []
    etypes = []
    for etype in g.canonical_etypes:
        src_type, _, dst_type = etype
        etype_id = g.get_etype_id(etype)
        src_type_id = g.get_ntype_id(src_type)
        dst_type_id = g.get_ntype_id(dst_type)
        g_sub = g[etype]
        indptr, indices, _ = g_sub.adj_sparse(fmt="csc")

        unique_nodes = th.nonzero(indptr[:-1] != indptr[1:]).squeeze(1)
        start = 0
        node_groups = []
        threshold = 0
        for end in range(0, len(unique_nodes)):
            indptr_val = indptr[unique_nodes[end]].item()
            if indptr_val >= threshold:
                node_groups.append(unique_nodes[start:end])
                start = end
                threshold += bucket_size
                etypes.append(th.LongTensor([etype_id]))
        node_groups.append(unique_nodes[start:])
        etypes.append(th.LongTensor([etype_id]))

        for node_group in node_groups:
            indptr_i.append(th.LongTensor([len(node_group) + indptr_i[-1].item()]))
            indices_i.append(node_group + g.ntype_pointer[dst_type_id])
            indptr_j.append(indptr[node_group + 1] + g.etype_pointer[etype_id])

        indices_j.append(indices + g.ntype_pointer[src_type_id])

    group_size = len(indptr_i) - 1
    etypes = tvm.nd.array(th.cat(etypes).numpy().astype("int32"), device=tvm.cuda(0))
    indptr_i = tvm.nd.array(th.cat(indptr_i).numpy().astype("int32"), device=tvm.cuda(0))
    indices_i = tvm.nd.array(th.cat(indices_i).numpy().astype("int32"), device=tvm.cuda(0))
    indptr_j = tvm.nd.array(th.cat(indptr_j).numpy().astype("int32"), device=tvm.cuda(0))
    indices_j = tvm.nd.array(th.cat(indices_j).numpy().astype("int32"), device=tvm.cuda(0))

    nnz_i = indices_i.shape[0]
    mod = tvm.IRModule.from_expr(
        rgcn_hetero_forward_2.specialize(
            {N: n, R: r, GROUP: group_size, FEAT_SIZE: feat_size, NNZ_I: nnz_i, NNZ_J: nnz_j}
        )
    )
    mod = lower_sparse_iter(mod)
    sch = tir.Schedule(mod)

    blk0 = sch.get_block("rgcn-hetero-forward0")
    blk1 = sch.get_block("rgcn-hetero-forward1")
    blk2 = sch.get_block("rgcn-hetero-forward2")
    read_blk = sch.cache_read(blk1, 2, "local")
    write_blk = sch.cache_write(blk2, 0, "local")
    sch.annotate(write_blk, "atomic", True)
    f_out, g = sch.get_loops(blk0)
    f_out_o, f_out_i = sch.split(f_out, [split_factor_f, None])
    (i,) = sch.get_loops(blk1)
    j, f_in = sch.get_loops(blk2)
    sch.bind(g, "blockIdx.y")
    sch.bind(f_out_o, "blockIdx.x")
    sch.bind(f_in, "threadIdx.x")
    sch.bind(f_out_i, "threadIdx.y")
    _, _, ax2 = sch.get_loops(read_blk)
    sch.bind(ax2, "threadIdx.x")
    mod = lower_sparse_buffer(sch.mod)
    f = tvm.build(mod["main"], target="cuda")

    cold_start = 3
    total = 10
    accum = 0

    for epoch in range(10):
        with TorchOpTimer() as timer:
            f(W, X, Y, etypes, indptr_i, indices_i, indptr_j, indices_j)
        if epoch == 0:
            tvm.testing.assert_allclose(Y.numpy(), ground_truth_y, rtol=1e-2)
        if epoch >= cold_start:
            accum += timer.time

    print("sparse-tir:\t\t {}ms".format(accum / (total - cold_start)))


if __name__ == "__main__":
    for feat_size in [32]:  # [4, 8, 16, 32, 64]:
        for name in ["am"]:  # ['aifb', 'mutag', 'bgs', 'am']:
            dataset = get_dataset_by_name(name)
            g = dataset[0]
            type_pointers = prepare_hetero_graph_simplified(g)
            n = g.num_nodes()
            r = len(g.etypes)
            feat = th.rand(n, feat_size).to(0) / 100
            weight = th.rand(r, feat_size, feat_size).to(0)
            # homograph
            g_homo = dgl.to_homogeneous(g)
            g_homo.ntype_pointer = type_pointers["ntype_node_pointer"]
            g_homo.etype_pointer = type_pointers["etype_edge_pointer"]
            g_homo.num_ntypes = max(g_homo.ndata[dgl.NTYPE]).item() + 1
            g_homo.num_rels = max(g_homo.edata[dgl.ETYPE]).item() + 1
            ground_truth_y = get_ground_truth(g_homo, feat, weight)
            # heterograph
            g.ntype_pointer = type_pointers["ntype_node_pointer"]
            g.etype_pointer = type_pointers["etype_edge_pointer"]
            for split_factor_f in [2, 4, 8, 16]:
                for bucket_size in [128, 256, 512, 1024, 2048]:
                    print(
                        "dataset {}, split_factor_f {}, bucket_size {}".format(
                            name, split_factor_f, bucket_size
                        )
                    )
                    test_lower_rgcn_hetero(
                        g,
                        feat_size,
                        feat.view(-1).cpu().numpy(),
                        weight.view(-1).cpu().numpy(),
                        ground_truth_y.view(-1).cpu().numpy(),
                        split_factor_f,
                        bucket_size,
                    )
