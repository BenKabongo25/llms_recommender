# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Linear aspects model: User profile

import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from utils import global_scale


class UserProfileType(enum.Enum):
    LINEAR = 0
    EMBED  = 1


class UserProfile:

    def __init__(self, n_users: int, args: Any):
        self.n_users = n_users
        self.args = args
        self.n_aspects = len(args.aspects)

    def get_users_parameters(self, U_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, U_ids: torch.Tensor, A_ratings: torch.Tensor) -> torch.Tensor:
        batch_users_params = self.get_users_parameters(U_ids)
        predictions = torch.sum(batch_users_params * A_ratings, dim=1)
        predictions = global_scale(predictions, self.args)
        return predictions


class LinearUserProfile(nn.Module, UserProfile):
    
    def __init__(self, n_users: int, args: Any):
        UserProfile.__init__(self, n_users, args)
        nn.Module.__init__(self)
        self.users_parameters = nn.Parameter(
            torch.full((self.n_users + 1, self.n_aspects), 1.0 / self.n_aspects)
        )

    def get_users_parameters(self, U_ids: torch.Tensor) -> torch.Tensor:
        user_params_normalized = F.normalize(self.users_parameters, p=1, dim=1)
        batch_users_params = user_params_normalized[U_ids]
        return batch_users_params
    
    def forward(self, U_ids: torch.Tensor, A_ratings: torch.Tensor) -> torch.Tensor:
        return UserProfile.forward(self, U_ids, A_ratings)


class EmbedUserProfile(nn.Module, UserProfile):
    
    def __init__(self, n_users: int, args: Any):
        UserProfile.__init__(self, n_users, args)
        nn.Module.__init__(self)
        self.users_embed = nn.Embedding(
            num_embeddings=self.n_users + 1,
            embedding_dim=self.args.embedding_dim,
            padding_idx=self.args.padding_idx
        )
        self.aspets_embed = nn.Embedding(
            num_embeddings=self.n_aspects + 1,
            embedding_dim=self.args.embedding_dim,
            padding_idx=self.args.padding_idx
        )

    def get_users_parameters(self, U_ids: torch.Tensor) -> torch.Tensor:
        users_embeddings = self.users_embed(U_ids)
        A_ids = torch.arange(1, self.n_aspects).to(U_ids.device)
        aspetcs_embeddings = self.aspets_embed(A_ids)
        weights = users_embeddings.dot(aspetcs_embeddings.T)
        return weights

    def forward(self, U_ids: torch.Tensor, A_ratings: torch.Tensor) -> torch.Tensor:
        return UserProfile.forward(self, U_ids, A_ratings)
