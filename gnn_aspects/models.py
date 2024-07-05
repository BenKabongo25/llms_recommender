
import torch
import torch.nn as nn
from torch_geometric.nn import (
    Sequential,
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
    

class GNNModel(nn.Module):

    def __init__(self, args):
        super(GNNModel, self).__init__()
        GNN = get_gnn(args.gnn_name)

        input_args = "x "
        if GNN in _EDGE_WEIGHT_GNNS:
            input_args += "edge_weight "
        if GNN in _EDGE_ATTR_GNNS:
            input_args += "edge_attr "

        gnn_input_args = input_args + "-> x"

        modules = []
        modules.append(
            (
                GNN(in_channels=1, out_channels=args.d_model, dropout=args.dropout),
                gnn_input_args
            )
        )
        for _ in range(args.n_layers - 2):
            modules.extend([
                (
                    GNN(in_channels=args.d_model, out_channels=args.d_model, dropout=args.dropout),
                    gnn_input_args
                ),
                nn.ReLU()
            ])
        modules.append(
            (
                GNN(in_channels=args.d_model, out_channels=args.d_model, dropout=args.dropout),
                gnn_input_args
            )
        )

        self.model = Sequential(input_args=input_args, modules=modules)
    
    def forward(self, data):
        return self.model(data)
    

class GNNAspectsModel(nn.Module):

    def __init__(self, args):
        super(GNNAspectsModel, self).__init__()
        
        self.user_gnn = GNNModel(args)
        self.item_gnn = GNNModel(args)

    def forward(self, user_data, item_data):
        user_x = self.user_gnn(user_data)
        item_x = self.item_gnn(item_data)

        rating = 
        
        return rating