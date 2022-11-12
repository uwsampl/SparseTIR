import dgl
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset


def get_hetero_dataset(name: str):
    if name == "aifb":
        return AIFBDataset()
    elif name == "mutag":
        return MUTAGDataset()
    elif name == "bgs":
        return BGSDataset()
    elif name == "am":
        return AMDataset()
    elif name == "biokg":
        return DglLinkPropPredDataset(name="ogbl-biokg")
    else:
        raise KeyError("Unknown dataset {}.".format(name))


def get_dataset(name: str):
    if name == "arxiv":
        arxiv = DglNodePropPredDataset(name="ogbn-arxiv")
        g = arxiv[0][0]
    elif name == "proteins":
        proteins = DglNodePropPredDataset(name="ogbn-proteins")
        g = proteins[0][0]
    elif name == "products":
        products = DglNodePropPredDataset(name="ogbn-products")
        g = products[0][0]
    elif name == "pubmed":
        pubmed = dgl.data.PubmedGraphDataset()
        g = pubmed[0]
    elif name == "citeseer":
        citeseer = dgl.data.CiteseerGraphDataset()
        g = citeseer[0]
    elif name == "cora":
        cora = dgl.data.CoraGraphDataset()
        g = cora[0]
    elif name == "ppi":
        ppi = dgl.data.PPIDataset()
        g = dgl.batch(ppi)
    elif name == "reddit":
        reddit = dgl.data.RedditDataset()
        g = reddit[0]
    elif name == "dense":
        g = dgl.rand_graph(1024, 1024 * 1024 // 2)
    else:
        raise KeyError("Unknown dataset {}.".format(name))
    return g.int()
