
import torch
import torch.nn as nn
from typing import Any

from user_profile import UserProfile
from item_profile import ItemProfile


class ABSARecommender(nn.Module):

    def __init__(self, user_profile: UserProfile, item_profile: ItemProfile, args: Any):
        nn.Module.__init__(self)
        self.user_profile = user_profile
        self.item_profile = item_profile
        self.args = args

    def forward(
        self,
        U_ids: torch.Tensor, 
        I_ids: torch.Tensor, 
        A_ratings: torch.Tensor=None,
        force: bool=True
    ) -> torch.Tensor:
        with torch.no_grad():
            U_params = self.user_profile.get_users_parameters(U_ids)
            I_params = self.item_profile.get_items_parameters(I_ids)
            if not force or A_ratings is None:
                used_A_ratings = I_params
            else:
                used_A_ratings = A_ratings.clone()
        U_R_hat = self.user_profile(U_ids, used_A_ratings)
        I_R_hat = self.item_profile(I_ids, U_params, A_ratings)
        return U_R_hat.squeeze(), I_R_hat.squeeze()


class RatingsMSELoss(nn.Module):

    def __init__(self, alpha: float=0.5):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "Alpha must be between 0 and 1!"
        self.alpha = alpha
        self.U_loss = nn.MSELoss(reduce="mean")
        self.I_loss = nn.MSELoss(reduce="mean")

    def forward(
        self, 
        U_R: torch.Tensor,
        U_R_hat: torch.Tensor,
        I_R: torch.Tensor,
        I_R_hat: torch.Tensor,
    ) -> torch.Tensor:
        #print(U_R.shape, U_R_hat.shape, I_R.shape, I_R_hat.shape)
        return (
            self.alpha * self.U_loss(U_R_hat.flatten(), U_R.flatten()) +
            (1 - self.alpha) * self.I_loss(I_R_hat.flatten(), I_R.flatten())
        )
