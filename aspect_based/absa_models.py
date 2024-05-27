# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Aspect-based profile approach
# Linear & deep models


import argparse
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from common.utils.evaluation import ratings_evaluation
from common.utils.vocabulary import Vocabulary


def rescale(x, a, b, c, d):
    return c + (c - d) * ((x - a) / (b - a))

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

    def __init__(self, data_df: pd.DataFrame, args):
        super().__init__()
        self.data_df = data_df
        self.args = args

    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, index) -> Any:
        row = self.data_df.iloc[index]
        user_id = row[self.args.user_vocab_id_column]
        global_rating = row[self.args.rating_column]
        aspects_ratings = [row[aspect] for aspect in self.args.aspects]
        return user_id, aspects_ratings, global_rating


class ABSARecommender:
    
    def __init__(self, n_users: int, args: Any):
        self.n_users = n_users
        self.args = args
        self.n_aspects = len(args.aspects)


class LinearABSARecommender(nn.Module, ABSARecommender):
    
    def __init__(self, n_users: int, args: Any):
        ABSARecommender.__init__(self, n_users, args)
        nn.Module.__init__(self)
        self.users_parameters = nn.Parameter(
            torch.full((self.n_users + 1, self.n_aspects), 1.0 / self.n_aspects)
        )

    def forward(self, U_ids: torch.Tensor, A_ratings: torch.Tensor) -> torch.Tensor:
        user_params_normalized = F.normalize(self.users_parameters, p=1, dim=1)
        batch_users_params = user_params_normalized[U_ids]
        predictions = torch.sum(batch_users_params * A_ratings.T, dim=1)
        predictions = global_scale(predictions, self.args)
        return predictions


class DeepABSARecommender(nn.Module, ABSARecommender):
    
    def __init__(self, n_users: int, args: Any):
        ABSARecommender.__init__(self, n_users, args)
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

    def forward(self, U_ids: torch.Tensor, A_ratings: torch.Tensor) -> torch.Tensor:
        users_embeddings = self.users_embed(U_ids)
        A_ids = torch.arange(1, self.n_aspects).to(U_ids.device)
        aspetcs_embeddings = self.aspets_embed(A_ids)
        weights = users_embeddings.dot(aspetcs_embeddings.T)
        predictions = torch.sum(weights * A_ratings, dim=1)
        predictions = global_scale(predictions, self.args)
        return predictions


def process_data(data_df: pd.DataFrame, args=None):
    users = data_df[args.user_id_column].unique()
    users_vocab = Vocabulary()
    users_vocab.add_elements(users)
    data_df[args.user_vocab_id_column] = data_df[args.user_id_column].apply(
        lambda u: users_vocab.element2id(u)
    )
    return data_df, users_vocab


def train(model, optimizer, dataloader, loss_fn, args):
    references = []
    predictions = []
    running_loss = .0

    model.train()
    for U_ids, A_ratings, R in dataloader:
        optimizer.zero_grad()
        U_ids = torch.LongTensor(U_ids).to(args.device)
        A_ratings = torch.stack(A_ratings, dim=0).to(dtype=torch.float32, device=args.device)
        R = torch.tensor(R, dtype=torch.float32).to(args.device)
        R_hat = model(U_ids, A_ratings).squeeze()
        loss = loss_fn(R_hat, R)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        references.extend(R.cpu().detach().tolist())
        predictions.extend(R_hat.cpu().detach().tolist())

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
        for U_ids, A_ratings, R in dataloader:
            U_ids = torch.LongTensor(U_ids).to(args.device)
            A_ratings = torch.stack(A_ratings, dim=0).to(dtype=torch.float32, device=args.device)
            R = torch.tensor(R, dtype=torch.float32).to(args.device)
            R_hat = model(U_ids, A_ratings).squeeze()
            loss = loss_fn(R_hat, R)
            running_loss += loss.item()

            references.extend(R.cpu().detach().tolist())
            predictions.extend(R_hat.cpu().detach().tolist())

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
    loss_fn = nn.MSELoss()

    train_infos = {
        "loss": [], "RMSE": [], "MAE": [], "P": [], "R": [], "F1": [], "AUC": []
    }
    test_infos = dict(train_infos)

    progress_bar = tqdm(range(1, 1 + args.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_epoch_infos = train(model, optimizer, train_dataloader, loss_fn, args)
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
    import numpy as np

    n_users = 10
    n_items = 20
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
    
    if isinstance(args.aspects, str):
        assert args.aspects.strip() != "", "You must specify aspects!"
        args.aspetcs = args.aspects.strip().split(args.aspects_sep)

    n_train = len(train_df)
    data_df = pd.concat([train_df, test_df])
    data_df, users_vocab = process_data(data_df, args)

    train_df = data_df.head(n_train)
    train_dataset = ABSARecoDataset(train_df, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_df = data_df.tail(len(data_df) - n_train)
    test_dataset = ABSARecoDataset(test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.linear:
        model = LinearABSARecommender(n_users=len(users_vocab), args=args)
    else:
        model = DeepABSARecommender(n_users=len(users_vocab), args=args)
    model.to(args.device)

    if args.exp_name == "":
        args.exp_name = f"absa_{('linear' if args.linear else 'deep')}_{int(time.time())}"
    exps_base_dir = os.path.join(args.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.log_file_path = os.path.join(exp_dir, "log.txt")
    args.res_file_path = os.path.join(exp_dir, "res.json")

    if args.verbose:
        log = (
            f"Task: Rating prediction\n" +
            f"Model: {('linear' if args.linear else 'deep')}\n" +
            f"Dataset: {args.dataset_name}\n" +
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
        plt.title(f"{args.dataset_name} MLP - {metric}")
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
    parser.add_argument("--exp_name", type=str, default="test")
    
    parser.add_argument("--aspects", type=str, default="")
    parser.add_argument("--aspects_sep", type=str, default=";")
    parser.add_argument("--linear", action=argparse.BooleanOptionalAction)
    parser.set_defaults(linear=True)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--padding_idx", type=int, default=0)

    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    parser.add_argument("--aspect_min_rating", type=float, default=1.0)
    parser.add_argument("--aspect_max_rating", type=float, default=5.0)
    
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.001)

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
