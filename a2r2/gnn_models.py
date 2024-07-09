# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# Attention and Aspect-based Rating and Review GNN Prediction Model 
# A2R2-v1-GNN & A2R2-v2-GNN Models


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

from models import AttentionLayer, get_mlp
from gnn_utils import get_gnn, has_edge_weight, has_edge_attr


class GNNModel(nn.Module):

    def __init__(self, args):
        super(GNNModel, self).__init__()
        self.args = args
        GNN = get_gnn(self.args.gnn_name)

        self._has_edge_weight = has_edge_weight(GNN)
        self._has_edge_attr = has_edge_attr(GNN)

        modules = []
        modules.append(GNN(in_channels=1, out_channels=self.args.embedding_dim))
        for _ in range(1, args.n_layers - 1):
            modules.append(GNN(in_channels=self.args.embedding_dim, out_channels=self.args.embedding_dim))
        modules.append(GNN(in_channels=self.args.embedding_dim, out_channels=self.args.embedding_dim))

        self.gnns = nn.ModuleList(modules)
    
    def forward(self, data):
        edge_weight = data.edge_weight
        edge_attr = None
        if self._has_edge_attr:
            edge_attr = edge_weight.unsqueeze(-1)

        kwargs = {"x": data.x, "edge_index": data.edge_index}
        if self._has_edge_weight:
            kwargs["edge_weight"] = edge_weight
        elif self._has_edge_attr:
            kwargs["edge_attr"] = edge_attr

        out = self.gnns[0](**kwargs)
        for i in range(1, self.args.n_layers):
            out = F.relu(out)
            kwargs["x"] = out
            out = self.gnns[i](**kwargs)
        return out


class A2R2GNN(object):

    def __init__(self, n_aspects: int, args: Any):
        super().__init__()

        self.n_aspects = n_aspects
        self.args = args
        assert (
            not self.args.do_classification or 
            (self.args.n_overall_ratings_classes > 1 and self.args.n_aspect_ratings_classes > 1)
        )

    def save(self, save_model_path: str):
        torch.save(self.state_dict(), save_model_path)

    def load(self, save_model_path: str):
        self.load_state_dict(torch.load(save_model_path))



class A2R2v1GNN(A2R2GNN, nn.Module):

    def __init__(self, n_aspects: int, args: Any):
        A2R2GNN.__init__(self, n_aspects, args)
        nn.Module.__init__(self)

        self.users_gnn = GNNModel(args)
        self.items_gnn = GNNModel(args)

        self.attn_u = AttentionLayer(self.args)

        if self.args.mlp_ratings_flag:
            self.mlp_overall_rating = get_mlp(
                in_features=2 * self.args.embedding_dim,
                n_classes=self.args.n_overall_ratings_classes
            )

            if self.args.mlp_aspect_shared_flag:
                self.mlp_aspect_rating = get_mlp(
                    in_features=2 * self.args.embedding_dim,
                    n_classes=self.args.n_aspect_ratings_classes
                )   
            else:
                self.mlp_aspect_rating = nn.ModuleList([
                    get_mlp(
                        in_features=2 * self.args.embedding_dim,
                        n_classes=self.args.n_aspect_ratings_classes
                    ) for _ in range(self.n_aspects)
                ])


    def forward(
        self,
        U_graphs: torch.Tensor,
        I_graphs: torch.Tensor,
    ) -> torch.Tensor:
        U_idx = torch.cumsum(
            torch.cat(
                [torch.zeros(1, device=self.args.device, dtype=int), U_graphs.y.flatten()[:-1]],
                dim=0
            ),
            dim=0
        )
        I_idx = torch.cumsum(
            torch.cat(
                [torch.zeros(1, device=self.args.device, dtype=int), I_graphs.y.flatten()[:-1]],
                dim=0
            ),
            dim=0
        )
        UA_idx = (
            U_idx.unsqueeze(1) + torch.arange(1, self.n_aspects + 1, device=self.args.device)
        ).flatten()
        IA_idx = (
            I_idx.unsqueeze(1) + torch.arange(1, self.n_aspects + 1, device=self.args.device)
        ).flatten()

        user_all_embeddings = self.users_gnn(U_graphs)
        item_all_embeddings = self.items_gnn(I_graphs)

        user_embeddings = user_all_embeddings[U_idx].unsqueeze(1)
        item_aspect_embeddings = item_all_embeddings[IA_idx].view(-1, self.n_aspects, self.args.embedding_dim)

        item_user_embeddings, user_attn = self.attn_u(user_embeddings, item_aspect_embeddings, item_aspect_embeddings)

        user_embeddings = user_embeddings.squeeze(1)
        item_user_embeddings = item_user_embeddings.squeeze(1)
        
        if self.args.mlp_ratings_flag:
            overall_rating = self.mlp_overall_rating(torch.cat([user_embeddings, item_user_embeddings], dim=-1))

            if self.args.mlp_aspect_shared_flag:
                user_aspect_embeddings = user_embeddings.unsqueeze(1).repeat(1, self.n_aspects, 1)
                user_aspect_embeddings = user_aspect_embeddings.view(-1, self.args.embedding_dim)
                item_aspect_embeddings = item_aspect_embeddings.view(-1, self.args.embedding_dim)
                aspect_ratings = self.mlp_aspect_rating(torch.cat([user_aspect_embeddings, item_aspect_embeddings], dim=-1))

            else:
                user_aspect_embeddings = user_embeddings.view(-1, self.args.embedding_dim)
                aspect_ratings = []
                for i in range(self.n_aspects):
                    aspect_embeddings = item_aspect_embeddings[:, i, :].view(-1, self.args.embedding_dim)
                    aspect_ratings_i = self.mlp_aspect_rating[i](
                        torch.cat([user_aspect_embeddings, aspect_embeddings], dim=-1)
                    ).unsqueeze(1)
                    aspect_ratings.append(aspect_ratings_i)
                aspect_ratings = torch.stack(aspect_ratings, dim=1)

            if self.args.do_classification:
                aspect_ratings = aspect_ratings.view(len(U_graphs), self.n_aspects, -1)
            else:
                aspect_ratings = aspect_ratings.view(-1, self.n_aspects)

        else:
            overall_rating = torch.einsum("bd,bd->b", user_embeddings, item_user_embeddings)
            aspect_ratings = torch.einsum("bd,bad->ba", user_embeddings, item_aspect_embeddings)

        return overall_rating, aspect_ratings, user_attn
        

class A2R2v2GNN(A2R2GNN, nn.Module):

    def __init__(self, n_aspects: int, args: Any):
        A2R2GNN.__init__(self, n_aspects, args)
        nn.Module.__init__(self)

        self.users_gnn = GNNModel(args)
        self.items_gnn = GNNModel(args)

        self.attn_u = AttentionLayer(self.args)
        self.attn_i = AttentionLayer(self.args)

        if self.args.mlp_ratings_flag:
            self.mlp_overall_rating = get_mlp(
                in_features=2 * self.args.embedding_dim,
                n_classes=self.args.n_overall_ratings_classes
            )
            
            if self.args.mlp_aspect_shared_flag:
                self.mlp_aspect_rating = get_mlp(
                    in_features=2 * self.args.embedding_dim,
                    n_classes=self.args.n_aspect_ratings_classes
                )   
            else:
                self.mlp_aspect_rating = nn.ModuleList([
                    get_mlp(
                        in_features=2 * self.args.embedding_dim,
                        n_classes=self.args.n_aspect_ratings_classes
                    ) for _ in range(self.n_aspects)
                ])


    def forward(
        self,
        U_graphs: torch.Tensor,
        I_graphs: torch.Tensor,
    ) -> torch.Tensor:
        U_idx = torch.cumsum(
            torch.cat(
                [torch.zeros(1, device=self.args.device), U_graphs.y.flatten()[:-1]],
                dim=0
            ),
            dim=0
        )
        I_idx = torch.cumsum(
            torch.cat(
                [torch.zeros(1, device=self.args.device), I_graphs.y.flatten()[:-1]],
                dim=0
            ),
            dim=0
        )
        UA_idx = (
            U_idx.unsqueeze(1) + torch.arange(1, self.n_aspects + 1, device=self.args.device)
        ).flatten()
        IA_idx = (
            I_idx.unsqueeze(1) + torch.arange(1, self.n_aspects + 1, device=self.args.device)
        ).flatten()

        user_all_embeddings = self.users_gnn(U_graphs)
        item_all_embeddings = self.items_gnn(I_graphs)

        user_embeddings = user_all_embeddings[U_idx].unsqueeze(1)
        user_aspect_embeddings = user_all_embeddings[UA_idx].view(-1, self.n_aspects, self.args.embedding_dim)

        item_embeddings = item_all_embeddings[I_idx].unsqueeze(1)
        item_aspect_embeddings = item_all_embeddings[IA_idx].view(-1, self.n_aspects, self.args.embedding_dim)

        item_user_embeddings, user_attn = self.attn_i(item_embeddings, user_aspect_embeddings, item_aspect_embeddings)
        user_item_embeddings, item_attn = self.attn_u(user_embeddings, item_aspect_embeddings, user_aspect_embeddings)

        user_embeddings = user_embeddings.squeeze(1)
        item_embeddings = item_embeddings.squeeze(1)
        user_item_embeddings = user_item_embeddings.squeeze(1)
        item_user_embeddings = item_user_embeddings.squeeze(1)
        
        if self.args.mlp_ratings_flag:
            overall_rating = self.mlp_overall_rating(torch.cat([user_item_embeddings, item_user_embeddings], dim=-1))

            if self.args.mlp_aspect_shared_flag:
                user_aspect_embeddings = user_aspect_embeddings.view(-1, self.args.embedding_dim)
                item_aspect_embeddings = item_aspect_embeddings.view(-1, self.args.embedding_dim)
                aspect_ratings = self.mlp_aspect_rating(torch.cat([user_aspect_embeddings, item_aspect_embeddings], dim=-1))

            else:
                aspect_ratings = []
                for i in range(self.n_aspects):
                    u_aspect_embeddings = user_aspect_embeddings[:, i, :].view(-1, self.args.embedding_dim)
                    i_aspect_embeddings = item_aspect_embeddings[:, i, :].view(-1, self.args.embedding_dim)
                    aspect_ratings_i = self.mlp_aspect_rating[i](
                        torch.cat([u_aspect_embeddings, i_aspect_embeddings], dim=-1)
                    ).unsqueeze(1)
                    aspect_ratings.append(aspect_ratings_i)
                aspect_ratings = torch.stack(aspect_ratings, dim=1)

            if self.args.do_classification:
                aspect_ratings = aspect_ratings.view(len(U_graphs), self.n_aspects, -1)
            else:
                aspect_ratings = aspect_ratings.view(-1, self.n_aspects)

        else:
            overall_rating = torch.einsum("bd,bd->b", user_item_embeddings, item_user_embeddings)
            aspect_ratings = torch.einsum("bad,bad->ba", user_aspect_embeddings, item_aspect_embeddings)

        return overall_rating, aspect_ratings, (user_attn, item_attn)
    
