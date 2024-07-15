# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# Attention and Aspect-based Rating and Review Prediction Model v1
# A2R2-v1 & A2R2-v2 Models


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


class AttentionLayer(nn.Module):

    def __init__(self, args: Any):
        super().__init__()

        self.args = args

        self.Wq = nn.Linear(self.args.embedding_dim, self.args.embedding_dim)
        self.Wk = nn.Linear(self.args.embedding_dim, self.args.embedding_dim)
        self.Wv = nn.Linear(self.args.embedding_dim, self.args.embedding_dim)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        attention = F.softmax(
            torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.args.embedding_dim)),
            dim=-1
        )

        return torch.matmul(attention, V), attention
    

def get_num_aspects_embeddings(n_elements: int, n_aspects: int) -> int:
    return n_elements * n_aspects + 1


def get_aspects_ids(Ids: torch.Tensor, n_aspects: int) -> torch.Tensor:
    A_ids = (Ids - 1).unsqueeze(1) * n_aspects + torch.arange(n_aspects).to(Ids.device) + 1
    A_ids[A_ids < 0] = 0
    return A_ids


def get_mlp(in_features: int, n_classes: int, factor: int=4) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, in_features // 2),
        nn.ReLU(),
        nn.Linear(in_features // 2, n_classes)
    )


class A2R2(object):

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_aspects: int,
        args: Any
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items 
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


class A2R2v1(A2R2, nn.Module):

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_aspects: int,
        args: Any
    ):
        A2R2.__init__(self, n_users, n_items, n_aspects, args)
        nn.Module.__init__(self)

        self.users_embed = nn.Embedding(
            num_embeddings=self.n_users + 1,
            embedding_dim=self.args.embedding_dim,
            padding_idx=self.args.padding_idx
        )

        self.items_aspects_embed = nn.Embedding(
            num_embeddings=get_num_aspects_embeddings(self.n_items, self.n_aspects),
            embedding_dim=self.args.embedding_dim,
            padding_idx=self.args.padding_idx
        )

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
        U_ids: torch.Tensor,
        I_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = self.users_embed(U_ids).unsqueeze(1)
        IA_ids = get_aspects_ids(I_ids, self.n_aspects)
        item_aspect_embeddings = self.items_aspects_embed(IA_ids)
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
                aspect_ratings = aspect_ratings.view(len(U_ids), self.n_aspects, -1)
            else:
                aspect_ratings = aspect_ratings.view(-1, self.n_aspects)

        else:
            overall_rating = torch.einsum("bd,bd->b", user_embeddings, item_user_embeddings)
            aspect_ratings = torch.einsum("bd,bad->ba", user_embeddings, item_aspect_embeddings)

        return overall_rating, aspect_ratings, user_attn

    def get_regularized_parameters(self) -> list:
        return [
            self.users_embed.weight,
            self.items_aspects_embed.weight
        ]
    

class A2R2v2(A2R2, nn.Module):

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_aspects: int,
        args: Any,
    ):
        A2R2.__init__(self, n_users, n_items, n_aspects, args)
        nn.Module.__init__(self)

        self.users_embed = nn.Embedding(
            num_embeddings=self.n_users + 1,
            embedding_dim=self.args.embedding_dim,
            padding_idx=self.args.padding_idx
        )

        self.users_aspects_embed = nn.Embedding(
            num_embeddings=get_num_aspects_embeddings(self.n_users, self.n_aspects),
            embedding_dim=self.args.embedding_dim,
            padding_idx=self.args.padding_idx
        )

        self.items_embed = nn.Embedding(
            num_embeddings=self.n_items + 1,
            embedding_dim=self.args.embedding_dim,
            padding_idx=self.args.padding_idx
        )

        self.items_aspects_embed = nn.Embedding(
            num_embeddings=get_num_aspects_embeddings(self.n_items, self.n_aspects),
            embedding_dim=self.args.embedding_dim,
            padding_idx=self.args.padding_idx
        )

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
        U_ids: torch.Tensor,
        I_ids: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = self.users_embed(U_ids).unsqueeze(1)
        UA_ids = get_aspects_ids(U_ids, self.n_aspects)
        user_aspect_embeddings = self.users_aspects_embed(UA_ids)

        item_embeddings = self.items_embed(I_ids).unsqueeze(1)
        IA_ids = get_aspects_ids(I_ids, self.n_aspects)
        item_aspect_embeddings = self.items_aspects_embed(IA_ids)

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
                aspect_ratings = aspect_ratings.view(len(U_ids), self.n_aspects, -1)
            else:
                aspect_ratings = aspect_ratings.view(-1, self.n_aspects)

        else:
            overall_rating = torch.einsum("bd,bd->b", user_item_embeddings, item_user_embeddings)
            aspect_ratings = torch.einsum("bad,bad->ba", user_aspect_embeddings, item_aspect_embeddings)

        return overall_rating, aspect_ratings, (user_attn, item_attn)

    
    def get_regularized_parameters(self) -> list:
        return [
            self.users_embed.weight,
            self.users_aspects_embed.weight,
            self.items_embed.weight,
            self.items_aspects_embed.weight
        ]


class RatingsLoss(nn.Module):

    def __init__(self, args: Any):
        super().__init__()
        self.args = args

        if self.args.do_classification:
            self.overall_rating_loss = nn.CrossEntropyLoss(ignore_index=self.args.padding_idx)
            self.aspect_rating_loss = nn.CrossEntropyLoss(ignore_index=self.args.padding_idx)
        else:
            self.overall_rating_loss = nn.MSELoss()
            self.aspect_rating_loss = nn.MSELoss()

    def forward(
        self, 
        R: torch.Tensor,
        R_hat: torch.Tensor,
        A_ratings: torch.Tensor,
        A_ratings_hat: torch.Tensor,
        *params_list: list,
    ) -> Tuple[torch.Tensor]:
        if self.args.do_classification:
            # R_hat: (batch_size, n_classes) ; A_ratings_hat: (batch_size, n_aspects, n_classes)
            R = R.long()
            A_ratings = A_ratings.long()
            overall_loss = self.overall_rating_loss(R_hat, R)
            aspect_loss = self.aspect_rating_loss(
                A_ratings_hat.view(-1, A_ratings_hat.size(-1)), 
                A_ratings.view(-1)
            )
        else:
            # R_hat: (batch_size) ; A_ratings_hat: (batch_size, n_aspects)
            overall_loss = self.overall_rating_loss(R_hat, R)
            aspect_loss = self.aspect_rating_loss(A_ratings_hat.flatten(), A_ratings.flatten())
        reg_loss = .0
        for params in params_list:
            reg_loss += torch.norm(params)
        return (
            self.args.alpha * overall_loss + self.args.beta * aspect_loss + self.args.lambda_ * reg_loss, 
            overall_loss, aspect_loss
        )
