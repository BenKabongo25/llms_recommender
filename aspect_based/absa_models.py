# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Aspect-based profile approach
# Linear & deep models


import argparse
import enum
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
warnings.filterwarnings(action="ignore")

from common.utils.evaluation import ratings_evaluation
from common.utils.vocabulary import Vocabulary


def rescale(x, a, b, c, d):
    return c + (d - c) * ((x - a) / (b - a))

aspect_scale = lambda x, args: rescale(
    x,
    args.min_rating, args.max_rating,
    args.aspect_min_rating, args.aspect_max_rating
)

global_scale = lambda x, args: rescale(
    x,
    args.aspect_min_rating, args.aspect_max_rating,
    args.min_rating, args.max_rating
)


class ABSARecoDataset(Dataset):

    def __init__(self, data_df: pd.DataFrame, args: Any):
        super().__init__()
        self.data_df = data_df
        self.args = args

    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, index) -> Any:
        row = self.data_df.iloc[index]
        user_id = row[self.args.user_vocab_id_column]
        item_id = row[self.args.item_vocab_id_column]
        global_rating = row[self.args.rating_column]
        aspects_ratings = [row[aspect] for aspect in self.args.aspects]
        return user_id, item_id, aspects_ratings, global_rating


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
        return  torch.clamp(
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


def process_data(data_df: pd.DataFrame, args: Any=None):
    users = data_df[args.user_id_column].unique()
    users_vocab = Vocabulary()
    users_vocab.add_elements(users)

    items = data_df[args.item_id_column].unique()
    items_vocab = Vocabulary()
    items_vocab.add_elements(items)

    def to_vocab_id(element, vocabulary: Vocabulary) -> int:
        return vocabulary.element2id(element)
    
    data_df[args.user_vocab_id_column] = data_df[args.user_id_column].apply(
        lambda u: to_vocab_id(u, users_vocab)
    )
    data_df[args.item_vocab_id_column] = data_df[args.item_id_column].apply(
        lambda i: to_vocab_id(i, items_vocab)
    )
    return data_df, users_vocab, items_vocab


def train(model, optimizer, dataloader, loss_fn, force, args):
    references = []
    predictions = []
    running_loss = .0

    model.train()
    for U_ids, I_ids, A_ratings, R in dataloader:
        U_ids = torch.LongTensor(U_ids).to(args.device)
        I_ids = torch.LongTensor(I_ids).to(args.device)
        A_ratings = torch.stack(A_ratings, dim=1).to(dtype=torch.float32, device=args.device)
        R = torch.tensor(R, dtype=torch.float32).to(args.device)
  
        optimizer.zero_grad()
        U_R_hat, I_R_hat = model(U_ids, I_ids, A_ratings, force)
        loss = loss_fn(R, U_R_hat, R, I_R_hat)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        references.extend(R.cpu().detach().tolist())
        predictions.extend(U_R_hat.cpu().detach().tolist())

    running_loss /= len(dataloader)
    ratings_scores = ratings_evaluation(predictions, references, args)

    return {
        "loss": running_loss, 
        "RMSE": ratings_scores["rmse"], 
        "MAE": ratings_scores["mae"], 
        "P": ratings_scores["precision"], 
        "R": ratings_scores["recall"], 
        "F1": ratings_scores["f1"], 
        "AUC": ratings_scores["auc"]
    }


def test(model, dataloader, loss_fn, args):
    references = []
    predictions = []
    running_loss = .0

    model.eval()
    with torch.no_grad():
        for U_ids, I_ids, A_ratings, R in dataloader:
            U_ids = torch.LongTensor(U_ids).to(args.device)
            I_ids = torch.LongTensor(I_ids).to(args.device)
            A_ratings = torch.stack(A_ratings, dim=1).to(dtype=torch.float32, device=args.device)
            R = torch.tensor(R, dtype=torch.float32).to(args.device)
    
            U_R_hat, I_R_hat = model(U_ids, I_ids, A_ratings, False)
            loss = loss_fn(R, U_R_hat, R, I_R_hat)
            running_loss += loss.item()

            references.extend(R.cpu().detach().tolist())
            predictions.extend(U_R_hat.cpu().detach().tolist())

    running_loss /= len(dataloader)
    ratings_scores = ratings_evaluation(predictions, references, args)

    return {
        "loss": running_loss, 
        "RMSE": ratings_scores["rmse"], 
        "MAE": ratings_scores["mae"], 
        "P": ratings_scores["precision"], 
        "R": ratings_scores["recall"], 
        "F1": ratings_scores["f1"], 
        "AUC": ratings_scores["auc"]
    }


def trainer(model, train_dataloader, test_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = RatingsMSELoss(args.alpha)

    train_infos = {
        "loss": [], "RMSE": [], "MAE": [], "P": [], "R": [], "F1": [], "AUC": []
    }
    test_infos = {
        "loss": [], "RMSE": [], "MAE": [], "P": [], "R": [], "F1": [], "AUC": []
    }

    progress_bar = tqdm(range(1, 1 + args.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        force_ratio = (
            args.force_initial_ratio - 
            (args.force_initial_ratio - args.force_final_ratio) *
            (epoch / args.n_epochs)
        )
        force = random.random() < force_ratio
        train_epoch_infos = train(model, optimizer, train_dataloader, loss_fn, force, args)
        test_epoch_infos = test(model, test_dataloader, loss_fn, args)
        for metric in train_infos:
            train_infos[metric].append(train_epoch_infos[metric])
            test_infos[metric].append(test_epoch_infos[metric])
        progress_bar.set_description(
            f"[{epoch} / {args.n_epochs}] " +
            f"RMSE: train={train_epoch_infos['RMSE']:.4f} test={test_epoch_infos['RMSE']:.4f} " +
            f"Loss: train={train_epoch_infos['loss']:.4f} test={test_epoch_infos['loss']:.4f}"
        )

        results = {"train": train_infos, "test": test_infos}
        with open(args.res_file_path, "w") as res_file:
            json.dump(results, res_file)

    return train_infos, test_infos


def make_toy_dataset():
    n_users = 100
    n_items = 200
    n_aspects = 5
    
    users = [f"u{i}" for i in range(1, n_users + 1)]
    items = [f"i{i}" for i in range(1, n_items + 1)]
    aspects = [f"a{i}" for i in range(1, n_aspects + 1)]

    users_params = np.random.random(size=(n_users, n_aspects))
    users_params = users_params / users_params.sum(axis=1)[:, np.newaxis]

    aspects_ratings = np.random.randint(1, 6, (n_users, n_items, n_aspects))

    data = []
    for ui, u in enumerate(users):
        for ii, i in enumerate(items):
            row = {}
            row["user_id"] = u
            row["item_id"] = i
            r_ui = 0
            for ai, a in enumerate(aspects):
                row[a] = aspects_ratings[ui, ii, ai]
                r_ui += users_params[ui, ai] * aspects_ratings[ui, ii, ai]
            row["rating"] = r_ui
            data.append(row)
    data_df = pd.DataFrame(data)

    return data_df, aspects


def main(args):
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)
    if args.dataset_path == "":
        args.dataset_path = os.path.join(args.dataset_dir, "data.csv")
        
    if args.dataset_name == "toy": # Toy dataset test
        data_df, aspetcs = make_toy_dataset()
        args.aspects = aspetcs
        train_df = data_df.sample(frac=args.train_size, random_state=args.random_state)
        test_df = data_df.drop(train_df.index)
    elif args.train_dataset_path != "" and args.test_dataset_path != "":
        train_df = pd.read_csv(args.train_dataset_path, index_col=0).dropna()
        test_df = pd.read_csv(args.test_dataset_path, index_col=0).dropna()
    elif args.dataset_path != "":
        data_df = pd.read_csv(args.dataset_path, index_col=0).dropna()
        train_df = data_df.sample(frac=args.train_size, random_state=args.random_state)
        test_df = data_df.drop(train_df.index)
    else:
        raise Exception("You must specify dataset_path or train/test data paths!")
    
    if not isinstance(args.aspects, list):
        assert args.aspects.strip() != "", "You must specify aspects!"
        args.aspetcs = args.aspects.strip().split(args.aspects_sep)

    n_train = len(train_df)
    data_df = pd.concat([train_df, test_df])
    data_df, users_vocab, items_vocab = process_data(data_df, args)
    n_users = len(users_vocab) + 1
    n_items = len(items_vocab) + 1

    train_df = data_df.head(n_train)
    train_dataset = ABSARecoDataset(train_df, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_df = data_df.tail(len(data_df) - n_train)
    test_dataset = ABSARecoDataset(test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    user_profile_type = UserProfileType(args.user_profile_type)
    if user_profile_type is UserProfileType.EMBED:
        user_profile = EmbedUserProfile(n_users, args)
    else: # LINEAR
        user_profile = LinearUserProfile(n_users, args)

    item_profile_type = ItemProfileType(args.item_profile_type)
    if item_profile_type is ItemProfileType.LEARNED:
        item_profile = LearnableItemProfile(n_items, args)
    else: # AVERAGE
        item_profile = AverageItemProfile(n_items, args)
    
    model = ABSARecommender(user_profile, item_profile, args)
    model.to(args.device)

    if args.exp_name == "":
        args.exp_name = (
            f"absa_{args.user_profile_type}_{args.item_profile_type}_"
            f"{int(time.time())}"
        )
    exps_base_dir = os.path.join(args.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.log_file_path = os.path.join(exp_dir, "log.txt")
    args.res_file_path = os.path.join(exp_dir, "res.json")

    if args.verbose:
        log = (
            f"Task: Rating prediction\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"User profile type: {args.user_profile_type}\n" +
            f"Item profile type: {args.item_profile_type}\n" +
            f"Device: {device}\n\n" +
            f"Data:\n{data_df.head(5)}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    train_infos, test_infos = trainer(model, train_dataloader, test_dataloader, args)
    results = {"train": train_infos, "test": test_infos}
    with open(args.res_file_path, "w") as res_file:
        json.dump(results, res_file)

    for metric in train_infos:
        plt.figure()
        plt.title(f"{args.dataset_name} ABSA model - {metric}")
        plt.plot(train_infos[metric], label="train")
        plt.plot(test_infos[metric], label="test")
        plt.legend()
        plt.savefig(os.path.join(exp_dir, metric.lower() + ".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, default="..\\Datasets\\AmazonReviews2023_process")
    parser.add_argument("--dataset_name", type=str, default="toy")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    
    parser.add_argument("--aspects", type=str, default="")
    parser.add_argument("--aspects_sep", type=str, default=";")
    parser.add_argument("--user_profile_type", type=int, default=0)
    parser.add_argument("--item_profile_type", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--padding_idx", type=int, default=0)

    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    parser.add_argument("--aspect_min_rating", type=float, default=1.0)
    parser.add_argument("--aspect_max_rating", type=float, default=5.0)
    
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--force_initial_ratio", type=float, default=1.0)
    parser.add_argument("--force_final_ratio", type=float, default=0.0)

    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")
    parser.add_argument("--user_vocab_id_column", type=str, default="user_vocab_id")
    parser.add_argument("--item_vocab_id_column", type=str, default="item_vocab_id")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.add_argument("--verbose_every", type=int, default=10)
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    main(args)
