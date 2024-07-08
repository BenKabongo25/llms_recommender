# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Classical RSs
# Neural Collaborative Filtering

import argparse
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
warnings.filterwarnings(action="ignore")

from common.utils.evaluation import ratings_evaluation
from common.utils.vocabulary import Vocabulary, create_vocab_from_df, to_vocab_id
from common.utils.functions import set_seed


class RatingDataset(Dataset):

    def __init__(self, ratings_df: pd.DataFrame, args):
        super().__init__()
        self.ratings_df = ratings_df
        self.args = args

    def __len__(self) -> int:
        return len(self.ratings_df)
    
    def __getitem__(self, index) -> Any:
        row = self.ratings_df.iloc[index]
        u = row[self.args.user_vocab_id_column]
        i = row[self.args.item_vocab_id_column]
        r = row[self.args.rating_column]
        return u, i, r


class MLPRecommender(nn.Module):

    def __init__(
            self, 
            n_users: int, 
            n_items: int, 
            embedding_dim: int=32,
            padding_idx: int=0,
            n_classes: int=1
        ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.n_classes = n_classes

        self.user_embed = nn.Embedding(
            num_embeddings=self.n_users + 1, 
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )
        self.item_embed = nn.Embedding(
            num_embeddings=self.n_items + 1, 
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )

        in_features = self.embedding_dim * 2
        out_features = self.embedding_dim
        min_out_features = max(2, self.n_classes) + 1
        hiddens_layers = []
        while out_features > min_out_features:
            hiddens_layers.append(nn.Linear(in_features, out_features))
            hiddens_layers.append(nn.ReLU())
            in_features = out_features
            out_features = out_features // 4
        output_layer = nn.Linear(in_features, self.n_classes)
        layers = [*hiddens_layers, output_layer]
        self.model = nn.Sequential(*layers)

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor) -> torch.Tensor:
        user_embeddings = self.user_embed(U_ids)
        item_embeddings = self.item_embed(I_ids)
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)
        out = self.model(embeddings)
        return out
    
    def save(self, save_model_path: str):
        torch.save(self.state_dict(), save_model_path)

    def load(self, save_model_path: str):
        self.load_state_dict(torch.load(save_model_path))


def train(model, optimizer, dataloader, loss_fn, args):
    references = []
    predictions = []
    running_loss = .0

    model.train()
    for U_ids, I_ids, R in dataloader:
        optimizer.zero_grad()
        U_ids = torch.LongTensor(U_ids).to(args.device)
        I_ids = torch.LongTensor(I_ids).to(args.device)
        if args.do_classification:
            R = torch.LongTensor(R).to(args.device)
        else:
            R = torch.tensor(R.clone().detach(), dtype=torch.float32).to(args.device)
        R_hat = model(U_ids, I_ids).squeeze()
        loss = loss_fn(R_hat, R)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if args.do_classification:
            R_hat = R_hat.argmax(dim=1)

        references.extend(R.cpu().detach().tolist())
        predictions.extend(R_hat.cpu().detach().tolist())

    running_loss /= len(dataloader)
    ratings_scores = ratings_evaluation(predictions, references, args)
    accuracy = -1
    if args.do_classification:
        accuracy = metrics.accuracy_score(references, predictions)

    return {
        "accuracy": accuracy, 
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
        for U_ids, I_ids, R in dataloader:
            U_ids = torch.LongTensor(U_ids).to(args.device)
            I_ids = torch.LongTensor(I_ids).to(args.device)
            if args.do_classification:
                R = torch.LongTensor(R).to(args.device)
            else:
                R = torch.tensor(R.clone().detach(), dtype=torch.float32).to(args.device)
            R_hat = model(U_ids, I_ids).squeeze()
            loss = loss_fn(R_hat, R)
            running_loss += loss.item()

            if args.do_classification:
                R_hat = R_hat.argmax(dim=1)

            references.extend(R.cpu().detach().tolist())
            predictions.extend(R_hat.cpu().detach().tolist())

    running_loss /= len(dataloader)
    ratings_scores = ratings_evaluation(predictions, references, args)
    accuracy = -1
    if args.do_classification:
        accuracy = metrics.accuracy_score(references, predictions)

    return {
        "accuracy": accuracy, 
        "loss": running_loss, 
        "RMSE": ratings_scores["rmse"], 
        "MAE": ratings_scores["mae"], 
        "P": ratings_scores["precision"], 
        "R": ratings_scores["recall"], 
        "F1": ratings_scores["f1"], 
        "AUC": ratings_scores["auc"]
    }


def get_loss_fn(args):
    if args.do_classification:
        return nn.CrossEntropyLoss(ignore_index=args.padding_idx)
    return nn.MSELoss()


def trainer(model, train_dataloader, test_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = get_loss_fn(args)

    train_infos = {
        "accuracy": [], "loss": [], "RMSE": [], "MAE": [], "P": [], "R": [], "F1": [], "AUC": []
    }
    test_infos = {
        "accuracy": [], "loss": [], "RMSE": [], "MAE": [], "P": [], "R": [], "F1": [], "AUC": []
    }

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

        if epoch % args.save_every == 0:
            model.save(args.save_model_path)

    return train_infos, test_infos


def get_vocabularies(args):
    metadata_dir = os.path.join(args.dataset_dir, "samples", "metadata")

    if "users_vocab.json" in os.listdir(metadata_dir):
        args.users_vocab_path = os.path.join(metadata_dir, "users_vocab.json")
    if args.users_vocab_path != "":
        users_vocab = Vocabulary()
        users_vocab.load(args.users_vocab_path)
    else:
        if args.users_path == "":
            args.users_path = os.path.join(metadata_dir, "users.csv")
        users_df = pd.read_csv(args.users_path)
        users_df[args.user_id_column] = users_df[args.user_id_column].apply(str)
        users_vocab = create_vocab_from_df(users_df, args.user_id_column)
        users_vocab.save(os.path.join(args.dataset_dir, "users_vocab.json"))

    if "items_vocab.json" in os.listdir(metadata_dir):
        args.items_vocab_path = os.path.join(metadata_dir, "items_vocab.json")
    if args.items_vocab_path != "":
        items_vocab = Vocabulary()
        items_vocab.load(args.items_vocab_path)
    else:
        if args.items_path == "":
            args.items_path = os.path.join(metadata_dir, "items.csv")
        items_df = pd.read_csv(args.items_path)
        items_df[args.item_id_column] = items_df[args.item_id_column].apply(str)
        items_vocab = create_vocab_from_df(items_df, args.item_id_column)
        items_vocab.save(os.path.join(args.dataset_dir, "items_vocab.json"))

    return users_vocab, items_vocab


def main_train_test(args):
    if args.dataset_path == "" and (args.train_dataset_path == "" or args.test_dataset_path == ""):
        seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
        args.train_dataset_path = os.path.join(seen_dir, "train.csv")
        args.test_dataset_path = os.path.join(seen_dir, "test.csv")

    if args.train_dataset_path != "" and args.test_dataset_path != "":
        train_df = pd.read_csv(args.train_dataset_path)
        test_df = pd.read_csv(args.test_dataset_path)
    else:
        data_df = pd.read_csv(args.dataset_path)
        data_df = data_df[[args.user_id_column, args.item_id_column, args.rating_column]]
        train_df = data_df.sample(frac=args.train_size, random_state=args.random_state)
        test_df = data_df.drop(train_df.index)

    train_df[args.user_id_column] = train_df[args.user_id_column].apply(str)
    train_df[args.item_id_column] = train_df[args.item_id_column].apply(str)
    test_df[args.user_id_column] = test_df[args.user_id_column].apply(str)
    test_df[args.item_id_column] = test_df[args.item_id_column].apply(str)
    
    train_df = train_df[[args.user_id_column, args.item_id_column, args.rating_column]]
    test_df = test_df[[args.user_id_column, args.item_id_column, args.rating_column]]

    users_vocab, items_vocab = get_vocabularies(args)

    train_df[args.user_vocab_id_column] = train_df[args.user_id_column].apply(
        lambda u: to_vocab_id(u, users_vocab)
    )
    test_df[args.user_vocab_id_column] = test_df[args.user_id_column].apply(
        lambda u: to_vocab_id(u, users_vocab)
    )

    train_df[args.item_vocab_id_column] = train_df[args.item_id_column].apply(
        lambda i: to_vocab_id(i, items_vocab)
    ) 
    test_df[args.item_vocab_id_column] = test_df[args.item_id_column].apply(
        lambda i: to_vocab_id(i, items_vocab)
    )
    
    n_classes = 1
    if args.do_classification:
        n_classes = int(args.max_rating - args.min_rating + 1)
        rating_fn = lambda r: int(r - args.min_rating)
        train_df[args.rating_column] = train_df[args.rating_column].apply(rating_fn)
        test_df[args.rating_column] = test_df[args.rating_column].apply(rating_fn)

    train_dataset = RatingDataset(train_df, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = RatingDataset(test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = MLPRecommender(
        n_users=len(users_vocab), 
        n_items=len(items_vocab), 
        embedding_dim=args.embedding_dim,
        padding_idx=args.padding_idx,
        n_classes=n_classes
    )
    model.to(args.device)
    if args.save_model_path != "":
        model.load(args.save_model_path)

    if args.exp_name == "":
        args.exp_name = f"mlp_{args.time_id}"
    exps_base_dir = os.path.join(args.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.log_file_path = os.path.join(exp_dir, "log.txt")
    args.res_file_path = os.path.join(exp_dir, "res.json")

    if args.save_model_path == "":
        args.save_model_path = os.path.join(exp_dir, "model.pth")

    if args.verbose:
        log = (
            f"Task: Rating prediction\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Device: {device}\n\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{train_df.head(5)}\n\n"
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


def main_test(args):
    if args.test_dataset_path == "":
        seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
        args.test_dataset_path = os.path.join(seen_dir, "test.csv")

    test_df = pd.read_csv(args.test_dataset_path)
    test_df[args.user_id_column] = test_df[args.user_id_column].apply(str)
    test_df[args.item_id_column] = test_df[args.item_id_column].apply(str)
    test_df = test_df[[args.user_id_column, args.item_id_column, args.rating_column]]

    users_vocab, items_vocab = get_vocabularies(args)

    test_df[args.user_vocab_id_column] = test_df[args.user_id_column].apply(
        lambda u: to_vocab_id(u, users_vocab)
    )
    test_df[args.item_vocab_id_column] = test_df[args.item_id_column].apply(
        lambda i: to_vocab_id(i, items_vocab)
    )
    
    n_classes = 1
    if args.do_classification:
        n_classes = int(args.max_rating - args.min_rating + 1)
        rating_fn = lambda r: int(r - args.min_rating)
        test_df[args.rating_column] = test_df[args.rating_column].apply(rating_fn)

    test_dataset = RatingDataset(test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = MLPRecommender(
        n_users=len(users_vocab), 
        n_items=len(items_vocab), 
        embedding_dim=args.embedding_dim,
        padding_idx=args.padding_idx,
        n_classes=n_classes
    )
    model.to(args.device)
    if args.save_model_path != "":
        model.load(args.save_model_path)

    if args.exp_name == "":
        args.exp_name = f"eval_mlp_{args.time_id}"
    exps_base_dir = os.path.join(args.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.log_file_path = os.path.join(exp_dir, "log.txt")
    args.res_file_path = os.path.join(exp_dir, "res.json")

    if args.verbose:
        log = (
            f"Task: Rating prediction\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Device: {device}\n\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{test_df.head(5)}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    loss_fn = get_loss_fn(args)
    test_results = test(model, test_dataloader, loss_fn, args)
    results = {"train": {}, "test": test_results}
    with open(args.res_file_path, "w") as res_file:
        json.dump(results, res_file)

    log = "Test Results:\n"
    for metric, value in test_results.items():
        log += f"{metric}: {value}\n"
    print(log)
    with open(args.log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(log)


def main(args):
    set_seed(args)

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)
    
    if args.train_flag:
        main_train_test(args)
    else:
        main_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, default="Datasets\\processed")
    parser.add_argument("--dataset_name", type=str, default="RateBeer")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--users_path", type=str, default="")
    parser.add_argument("--items_path", type=str, default="")
    parser.add_argument("--users_vocab_path", type=str, default="")
    parser.add_argument("--items_vocab_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="mlp")
    
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--padding_idx", type=int, default=0)
    parser.add_argument("--do_classification", action=argparse.BooleanOptionalAction)
    parser.set_defaults(do_classification=False)

    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.005)
    
    parser.add_argument("--train_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(train_flag=True)
    parser.add_argument("--save_model_path", type=str, default="")
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="mlp")
    parser.add_argument("--user_vocab_id_column", type=str, default="user_vocab_id")
    parser.add_argument("--item_vocab_id_column", type=str, default="item_vocab_id")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.add_argument("--verbose_every", type=int, default=10)
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    main(args)