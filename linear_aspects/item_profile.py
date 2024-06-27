# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Linear aspects model: Item profile

import enum
import torch
import torch.nn as nn
from typing import Any


class ItemProfileType(enum.Enum):
    AVERAGE     = 0
    WEIGHTED    = 1
    USER_BASED  = 2
    LEARNED     = 3


class ItemProfile:

    def __init__(self, n_items: int, args: Any):
        self.n_items = n_items
        self.args = args
        self.n_aspects = len(self.args.aspects)

    def get_items_parameters(self, I_ids: torch.Tensor) -> torch.Tensor:
        return self.clamp(self.items_parameters[I_ids])
    
    def clamp(self, parameters: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            parameters,
            min=self.args.aspect_min_rating,
            max=self.args.aspect_max_rating
        )


class AverageItemProfile(nn.Module, ItemProfile):

    def __init__(self, n_items: int, args: Any):
        ItemProfile.__init__(self, n_items, args)
        nn.Module.__init__(self)
        self.items_parameters = torch.zeros((self.n_items + 1, self.n_aspects))
        self.items_counters = torch.zeros(self.n_items + 1)

    def forward(
        self, 
        I_ids: torch.Tensor, 
        A_weights: torch.Tensor,
        A_ratings: torch.Tensor=None
    ) -> torch.Tensor:
        if A_ratings is not None:
            items_parameters = self.items_parameters.clone() * self.items_counters[:, None]
            items_parameters.index_add_(0, I_ids, A_ratings)
            counts = self.items_counters.clone()
            counts.index_add_(0, I_ids, torch.ones(len(I_ids)).to(I_ids.device))
            items_parameters = items_parameters / torch.clamp(counts, min=1)[:, None]
            self.items_parameters = items_parameters
        A_ratings = self.get_items_parameters(I_ids)
        predictions = torch.sum(A_weights * A_ratings, dim=1)
        return predictions
    

class LearnableItemProfile(nn.Module, ItemProfile):

    def __init__(self, n_items: int, args: Any):
        ItemProfile.__init__(self, n_items, args)
        nn.Module.__init__(self)
        self.items_parameters = nn.Parameter(
            torch.full(
                (self.n_items + 1, self.n_aspects),
                (self.args.aspect_min_rating + self.args.aspect_max_rating) / 2.0
            )
        )

    def forward(
        self, 
        I_ids: torch.Tensor, 
        A_weights: torch.Tensor,
        A_ratings: torch.Tensor=None
    ) -> torch.Tensor:
        A_ratings = self.get_items_parameters(I_ids)
        predictions = torch.sum(A_weights * A_ratings, dim=1)
        return predictions
