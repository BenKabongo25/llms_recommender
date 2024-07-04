# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# Attention and Aspect-based Rating and Review Prediction Model v1
# A2R2-v&


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *


class A2R2v1(nn.Module):

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

        self.users_embed = nn.Embedding(
            num_embeddings=self.n_users + 1,
            embedding_dim=self.args.embedding_dim,
            padding_idx=self.args.padding_idx
        )

        self.items_aspects_embed = nn.Embedding(
            num_embeddings=(self.n_items * self.n_aspects)s + 1,
            embedding_dim=self.args.embedding_dim,
            padding_idx=self.args.padding_idx
        )

        self.rating_pred = lambda u, i: torch.

