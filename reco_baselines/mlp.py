# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Classical RSs
# Neural Collaborative Filtering

import argparse
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from common.utils.evaluation import ratings_evaluation


class Vocabulary:

    def __init__(self):
        self._elements2ids = {}
        self._ids2elements = {}
        self.n_elements = 0

    def add_element(self, element: Union[int, float, str]):
        if element not in self._elements2ids:
            self.n_elements += 1
            self._elements2ids[element] = self.n_elements
            self._ids2elements[self.n_elements] = element

    def add_elements(self, elements: List[Union[int, float, str]]):
        for element in tqdm(elements, "Vocabulary creation", colour="green"):
            self.add_element(element)

    def __len__(self):
        return self.n_elements + 1
    
    def id2element(self, id: int) -> Union[int, float, str]:
        return self._ids2elements[id]
    
    def element2id(self, element: Union[int, float, str]) -> int:
        return self._elements2ids[element]
    
    def ids2elements(self, ids: List[int]) -> List[Union[int, float, str]]:
        return [self._ids2elements[id] for id in ids]
    
    def elements2ids(self, elements: List[Union[int, float, str]]) -> List[int]:
        return [self._elements2ids[element] for element in elements]


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
            num_embeddings=self.n_users, 
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )
        self.item_embed = nn.Embedding(
            num_embeddings=self.n_items, 
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
            out_features = out_features // 2
        output_layer = nn.Linear(in_features, self.n_classes)
        layers = [*hiddens_layers, output_layer]
        self.model = nn.Sequential(*layers)

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor) -> torch.Tensor:
        user_embeddings = self.user_embed(U_ids)
        item_embeddings = self.item_embed(I_ids)
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)
        out = self.model(embeddings)
        return out


def process_data(data_df: pd.DataFrame, args=None):
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
    if args.do_classification:
        data_df[args.rating_column] = data_df[args.rating_column].apply(int)
    return data_df, users_vocab, items_vocab


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
                R_hat = args.min_rating + R_hat
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


def trainer(model, train_dataloader, test_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.do_classification:
        loss_fn = nn.CrossEntropyLoss(ignore_index=args.padding_idx)
    else:
        loss_fn = nn.MSELoss()

    train_infos = {
        "accuracy": [], "loss": [], "RMSE": [], "MAE": [], "P": [], "R": [], "F1": [], "AUC": []
    }
    test_infos = dict(train_infos)

    for epoch in tqdm(range(1, 1 + args.n_epochs), "Training", colour="blue"):
        train_epoch_infos = train(model, optimizer, train_dataloader, loss_fn, args)
        test_epoch_infos = test(model, test_dataloader, loss_fn, args)
        for metric in train_infos:
            train_infos[metric].append(train_epoch_infos[metric])
            test_infos[metric].append(test_epoch_infos[metric])
        #print(test_infos)

    return train_infos, test_infos


def main(args):
    if args.dataset_path != "":
        data_df = pd.read_csv(args.dataset_path, index_col=0).dropna().sample(160)
        train_df = data_df.sample(frac=args.train_size)
        test_df = data_df.drop(train_df.index)
    elif args.train_dataset_path != "" and args.test_dataset_path != "":
        train_df = pd.read_csv(args.train_dataset_path, index_col=0)
        test_df = pd.read_csv(args.test_dataset_path, index_col=0)
    else:
        raise Exception("You must specifie dataset_path or train/test data paths!")
    
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    n_train = len(train_df)
    data_df = pd.concat([train_df, test_df])
    data_df, users_vocab, items_vocab = process_data(data_df, args)
    print(data_df.sample(n=5))

    train_df = data_df.head(n_train)
    train_dataset = RatingDataset(train_df, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_df = data_df.tail(len(data_df) - n_train)
    test_dataset = RatingDataset(test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    n_classes = 1
    if args.do_classification:
        n_classes = int(args.max_rating)

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

    train_infos, test_infos = trainer(model, train_dataloader, test_dataloader, args)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="All_Beauty")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--padding_idx", type=int, default=0)
    parser.add_argument("--do_classification", action=argparse.BooleanOptionalAction)
    parser.set_defaults(do_classification=False)

    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")
    parser.add_argument("--user_vocab_id_column", type=str, default="user_vocab_id")
    parser.add_argument("--item_vocab_id_column", type=str, default="item_vocab_id")
    parser.add_argument("--random_state'", type=int, default=42)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.add_argument("--verbose_every", type=int, default=10)
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    main(args)