# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# A2R2-GNN: Utils

import torch
import torch.nn as nn
from torch_geometric.nn import (
    SimpleConv,
    GCNConv,
    ChebConv,
    GraphConv,
    GatedGraphConv,
    TAGConv,
    ARMAConv,
    SGConv,
    SSGConv,
    APPNP,
    DNAConv,
    LEConv,
    GCN2Conv,
    WLConv,
    FAConv,
    LGConv,
    MixHopConv,
    ResGatedGraphConv,
    GATConv,
    GATv2Conv,
    TransformerConv,
    GINEConv,
    GMMConv,
    SplineConv,
    NNConv,
    CGConv,
    PNAConv,
    GENConv,
    PDNConv,
    GeneralConv
)


_EDGE_WEIGHT_GNNS = (
    SimpleConv,
    GCNConv,
    ChebConv,
    GraphConv,
    GatedGraphConv,
    TAGConv,
    ARMAConv,
    SGConv,
    SSGConv,
    APPNP,
    DNAConv,
    LEConv,
    GCN2Conv,
    WLConv,
    FAConv,
    LGConv,
    MixHopConv,
)

_EDGE_ATTR_GNNS = (
    ResGatedGraphConv,
    GATConv,
    GATv2Conv,
    TransformerConv,
    GINEConv,
    GMMConv,
    SplineConv,
    NNConv,
    CGConv,
    PNAConv,
    GENConv,
    PDNConv,
    GeneralConv
)


def get_gnn(gnn_name: str):
    if gnn_name == "SimpleConv":
        return SimpleConv
    elif gnn_name == "GCNConv":
        return GCNConv
    elif gnn_name == "ChebConv":
        return ChebConv
    elif gnn_name == "GraphConv":
        return GraphConv
    elif gnn_name == "GatedGraphConv":
        return GatedGraphConv
    elif gnn_name == "TAGConv":
        return TAGConv
    elif gnn_name == "ARMAConv":
        return ARMAConv
    elif gnn_name == "SGConv":
        return SGConv
    elif gnn_name == "SSGConv":
        return SSGConv
    elif gnn_name == "APPNP":
        return APPNP
    elif gnn_name == "DNAConv":
        return DNAConv
    elif gnn_name == "LEConv":
        return LEConv
    elif gnn_name == "GCN2Conv":
        return GCN2Conv
    elif gnn_name == "WLConv":
        return WLConv
    elif gnn_name == "FAConv":
        return FAConv
    elif gnn_name == "LGConv":
        return LGConv
    elif gnn_name == "MixHopConv":
        return MixHopConv
    elif gnn_name == "ResGatedGraphConv":
        return ResGatedGraphConv
    elif gnn_name == "GATConv":
        return GATConv
    elif gnn_name == "GATv2Conv":
        return GATv2Conv
    elif gnn_name == "TransformerConv":
        return TransformerConv
    elif gnn_name == "GINEConv":
        return GINEConv
    elif gnn_name == "GMMConv":
        return GMMConv
    elif gnn_name == "SplineConv":
        return SplineConv
    elif gnn_name == "NNConv":
        return NNConv
    elif gnn_name == "CGConv":
        return CGConv
    elif gnn_name == "PNAConv":
        return PNAConv
    elif gnn_name == "GENConv":
        return GENConv
    elif gnn_name == "PDNConv":
        return PDNConv
    elif gnn_name == "GeneralConv":
        return GeneralConv
    else:
        raise ValueError("Invalid GNN name")
    

def has_edge_weight(GNN):
    return GNN in _EDGE_WEIGHT_GNNS

def has_edge_attr(GNN):
    return GNN in _EDGE_ATTR_GNNS
