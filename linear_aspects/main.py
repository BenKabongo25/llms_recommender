# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Linear Aspect-based approach


import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import *

from data import *
from item_profile import *
from model import *
from user_profile import *
from utils import *


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
        **ratings_scores
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
        **ratings_scores
    }


def trainer(model, train_dataloader, test_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = RatingsMSELoss(args.alpha)

    train_infos = {}
    test_infos = {}

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
        
        for metric in train_epoch_infos:
            if metric not in train_infos:
                train_infos[metric] = []
                test_infos[metric] = []
            train_infos[metric].append(train_epoch_infos[metric])
            test_infos[metric].append(test_epoch_infos[metric])

        progress_bar.set_description(
            f"[{epoch} / {args.n_epochs}] " +
            f"RMSE: train={train_epoch_infos['rmse']:.4f} test={test_epoch_infos['rmse']:.4f} " +
            f"Loss: train={train_epoch_infos['loss']:.4f} test={test_epoch_infos['loss']:.4f}"
        )

        results = {"train": train_infos, "test": test_infos}
        with open(args.res_file_path, "w") as res_file:
            json.dump(results, res_file)

    return train_infos, test_infos


def main(args):
    set_seed(args)

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)

    train_df, test_df = get_train_test_data(args)
    
    assert args.aspects.strip() != "", "You must specify aspects!"
    args.aspects = args.aspects.strip().split(args.aspects_sep)

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
            f"Aspects: {args.aspects}\n" +
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

    parser.add_argument("--base_dir", type=str, default="Datasets\\processed")
    parser.add_argument("--dataset_name", type=str, default="TripAdvisor")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    
    parser.add_argument("--aspects", type=str, 
        default="service cleanliness value sleep_quality rooms location")
    parser.add_argument("--aspects_sep", type=str, default=" ")
    parser.add_argument("--user_profile_type", type=int, default=0)
    parser.add_argument("--item_profile_type", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--padding_idx", type=int, default=0)

    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    parser.add_argument("--aspect_min_rating", type=float, default=1.0)
    parser.add_argument("--aspect_max_rating", type=float, default=5.0)
    
    parser.add_argument("--n_epochs", type=int, default=30)
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
