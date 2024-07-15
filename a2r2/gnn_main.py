# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# A2R2-GNN: Main


import argparse
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import *

from data import *
from models import *
from utils import *
from gnn_data import *
from gnn_models import *
from gnn_utils import *


def train_test(model, optimizer, dataloader, loss_fn, args, train=True):
    references = {}
    predictions = {}
    for aspect in args.aspects + ["overall"]:
        references[aspect] = []
        predictions[aspect] = []

    running_loss = .0
    overall_running_loss = .0
    aspect_running_loss = .0

    if train:
        model.train()
    else:
        model.eval()

    for U_graphs, I_graphs, A_ratings, R in dataloader:
        U_graphs = U_graphs.to(args.device)
        I_graphs = I_graphs.to(args.device)
        A_ratings = torch.stack(A_ratings, dim=1).to(dtype=torch.float32, device=args.device)
        R = torch.tensor(R, dtype=torch.float32).to(args.device)
  
        R_hat, A_ratings_hat, attn = model(U_graphs, I_graphs)
        R_hat = R_hat.squeeze()
        A_ratings_hat = A_ratings_hat.squeeze()
        loss, overall_loss, aspect_loss = loss_fn(R, R_hat, A_ratings, A_ratings_hat)
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        overall_running_loss += overall_loss.item()
        aspect_running_loss += aspect_loss.item()

        if args.do_classification:
            R_hat = R_hat.argmax(dim=-1).squeeze()
            A_ratings_hat = A_ratings_hat.argmax(dim=-1).squeeze()

        references["overall"].extend(R.cpu().detach().tolist())
        predictions["overall"].extend(R_hat.cpu().detach().tolist())

        for i, aspect in enumerate(args.aspects):
            references[aspect].extend(A_ratings[:, i].cpu().detach().tolist())
            predictions[aspect].extend(A_ratings_hat[:, i].cpu().detach().tolist())

    running_loss /= len(dataloader)
    overall_running_loss /= len(dataloader)
    aspect_running_loss /= len(dataloader)
    loss = {"loss": running_loss, "overall": overall_running_loss, "aspect": aspect_running_loss}
    ratings_scores = ratings_aspects_evaluation(predictions, references, args)
    return {"loss": loss, **ratings_scores}


def trainer(model, train_dataloader, test_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = RatingsLoss(args)

    train_infos = {}
    test_infos = {}

    progress_bar = tqdm(range(1, 1 + args.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_epoch_infos = train_test(model, optimizer, train_dataloader, loss_fn, args, train=True)
        with torch.no_grad():
            test_epoch_infos = train_test(model, None, test_dataloader, loss_fn, args, train=False)
        
        for k_1 in train_epoch_infos.keys():
            if k_1 not in train_infos.keys():
                train_infos[k_1] = {}
                test_infos[k_1] = {}
            for k_2 in train_epoch_infos[k_1].keys():
                if k_2 not in train_infos[k_1].keys():
                    train_infos[k_1][k_2] = []
                    test_infos[k_1][k_2] = []
                train_infos[k_1][k_2].append(train_epoch_infos[k_1][k_2])
                test_infos[k_1][k_2].append(test_epoch_infos[k_1][k_2])

        train_overall_rmse = train_epoch_infos["overall"]["rmse"]
        test_overall_rmse = test_epoch_infos["overall"]["rmse"]
        train_loss = train_epoch_infos["loss"]["loss"]
        test_loss = test_epoch_infos["loss"]["loss"]

        progress_bar.set_description(
            f"[{epoch} / {args.n_epochs}] " +
            f"Rmse: train={train_overall_rmse:.4f} test={test_overall_rmse:.4f} " +
            f"Loss: train={train_loss:.4f} test={test_loss:.4f}"
        )

        results = {"train": train_infos, "test": test_infos}
        with open(args.res_file_path, "w") as res_file:
            json.dump(results, res_file)

        if epoch % args.save_every == 0:
            model.save(args.save_model_path)

        if args.verbose and epoch % args.verbose_every == 0:
            for k_1 in train_infos.keys():
                for k_2 in train_infos[k_1].keys():
                    plt.figure()
                    plt.title(f"{args.dataset_name} A2R2 - {k_1} - {k_2}")
                    plt.plot(train_infos[k_1][k_2], label="train")
                    plt.plot(test_infos[k_1][k_2], label="test")
                    img_name = f"{k_1}_{k_2}.png".lower()
                    plt.savefig(os.path.join(args.exp_dir, img_name))
                    plt.close()

    return train_infos, test_infos


def main_train_test(args):
    train_df, test_df = get_train_test_data(args)
    
    if args.do_classification:
        if args.n_overall_ratings_classes == 1:
            args.n_overall_ratings_classes = int(args.max_rating - args.min_rating + 1)
        overall_rating_fn = lambda r: int(r - args.min_rating)
        train_df[args.rating_column] = train_df[args.rating_column].apply(overall_rating_fn)
        test_df[args.rating_column] = test_df[args.rating_column].apply(overall_rating_fn)

        if args.n_aspect_ratings_classes == 1:
            args.n_aspect_ratings_classes = int(args.aspect_max_rating - args.aspect_min_rating + 1)
        aspect_rating_fn = lambda r: int(r - args.aspect_min_rating)
        for aspect in args.aspects:
            train_df[aspect] = train_df[aspect].apply(aspect_rating_fn)
            test_df[aspect] = test_df[aspect].apply(aspect_rating_fn)

    users_graphs, items_graphs = dataset2graphs(train_df, args)
    for j, (_, row) in tqdm(enumerate(test_df.iterrows()), "Test graphs check", colour="green"):
        user_id = row[args.user_id_column]
        item_id = row[args.item_id_column]
        if user_id not in users_graphs:
            users_graphs[user_id] = get_default_graph(args)
        if item_id not in items_graphs:
            items_graphs[item_id] = get_default_graph(args)

    train_dataset = ABSAGNNRecoDataset(users_graphs, items_graphs, train_df, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = ABSAGNNRecoDataset(users_graphs, items_graphs, test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if args.model_version == 2:
        model = A2R2v2GNN(len(args.aspects), args)
    else:
        args.model_version = 1
        model = A2R2v1GNN(len(args.aspects), args)
    model.to(args.device)
    print(model)
    if args.save_model_path != "":
        model.load(args.save_model_path)

    if args.exp_name == "":
        prediction_function = "mlp" if args.mlp_ratings_flag else "dotproduct"
        prediction_type = "classification" if args.do_classification else "regression"
        shared = "shared" if args.mlp_aspect_shared_flag else "separate"
        args.exp_name = (
            f"a2r2v{args.model_version}{args.gnn_name}_{prediction_function}_{prediction_type}_{shared}_"
            f"{int(time.time())}"
        )
    exps_base_dir = os.path.join(args.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.exp_dir = exp_dir
    args.log_file_path = os.path.join(exp_dir, "log.txt")
    args.res_file_path = os.path.join(exp_dir, "res.json")

    if args.save_model_path == "":
        args.save_model_path = os.path.join(exp_dir, "model.pth")

    if args.verbose:
        log = (
            f"Task: Rating prediction\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Model: A2R2v{args.model_version}GNN\n" +
            f"Aspects: {args.aspects}\n" +
            f"Device: {device}\n\n" +
            f"Args:\n{args}\n\n" +
            f"Data:\n{train_df.head(5)}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    train_infos, test_infos = trainer(model, train_dataloader, test_dataloader, args)
    results = {"train": train_infos, "test": test_infos}
    with open(args.res_file_path, "w") as res_file:
        json.dump(results, res_file)


def main_test(args):
    pass


def main(args):
    set_seed(args)

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)

    assert args.aspects.strip() != "", "You must specify aspects!"
    args.aspects = args.aspects.strip().split(args.aspects_sep)
    
    if args.train_flag:
        main_train_test(args)
    else:
        main_test(args)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, default=os.path.join("datasets", "processed"))
    parser.add_argument("--dataset_name", type=str, default="TripAdvisor")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--users_path", type=str, default="")
    parser.add_argument("--items_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    
    parser.add_argument("--aspects", type=str, 
        default="service cleanliness value sleep_quality rooms location")
        #default="appearance aroma taste palate")
    parser.add_argument("--aspects_sep", type=str, default=" ")

    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    parser.add_argument("--aspect_min_rating", type=float, default=1.0)
    parser.add_argument("--aspect_max_rating", type=float, default=5.0)
    
    parser.add_argument("--model_version", type=int, default=1)
    parser.add_argument("--gnn_name", type=str, default="GATConv")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_layers", type=int, default=3)

    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--mlp_ratings_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(mlp_ratings_flag=True)
    parser.add_argument("--mlp_aspect_shared_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(mlp_aspect_shared_flag=False)
    parser.add_argument("--do_classification", action=argparse.BooleanOptionalAction)
    parser.set_defaults(do_classification=False)
    parser.add_argument("--n_overall_ratings_classes", type=int, default=1)
    parser.add_argument("--n_aspect_ratings_classes", type=int, default=1)

    parser.add_argument("--train_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(train_flag=True)
    parser.add_argument("--save_model_path", type=str, default="")
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.add_argument("--verbose_every", type=int, default=5)
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
    main(args)
