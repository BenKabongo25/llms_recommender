# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# A2R2-GNN: Data

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm
from typing import Any, Dict, List, Tuple


def document2graph(documents_df: pd.DataFrame, user_based: bool=True, args: Any=None) -> Data:
    x = []
    node_name = []
    edge_index = []
    edge_weight = []

    x.append([0]) # user(item)
    node_name.append("u" if user_based else "i")

    for i in range(1, len(args.aspects) + 1):
        x.append([1]) # aspect
        aspect = args.aspects[i - 1]
        node_name.append(aspect)
        weight = (np.mean(documents_df[aspect]) - args.aspect_min_rating) / (args.aspect_max_rating - args.aspect_min_rating)
        edge_index.append([0, i])
        edge_index.append([i, 0])
        edge_weight.append(weight)
        edge_weight.append(weight)

    last_index = len(x)
    for j, (_, row) in enumerate(documents_df.iterrows()):
        x.append([2]) # item(user)
        node_name.append(("i" if user_based else "u") + str(j))

        for i in range(1, len(args.aspects) + 1):
            aspect = args.aspects[i - 1]
            rating = row[aspect]
            edge_index.append([i, j + last_index])
            edge_index.append([j + last_index, i])
            weight = (rating - args.aspect_min_rating) / (args.aspect_max_rating - args.aspect_min_rating)
            edge_weight.append(weight)
            edge_weight.append(weight)

    num_nodes = len(x)
    x = torch.FloatTensor(x)
    edge_index = torch.LongTensor(edge_index).T
    edge_weight = torch.FloatTensor(edge_weight)
    y = torch.LongTensor([num_nodes])

    return Data(
        num_nodes=num_nodes, 
        x=x, 
        y=y, 
        edge_index=edge_index, 
        edge_weight=edge_weight, 
        node_name=node_name
    )


def get_default_graph(args: Any) -> Data:
    x = []
    node_name = []
    edge_index = []
    edge_weight = []

    x.append([0]) # user(item)
    node_name.append(".")

    for i in range(1, len(args.aspects) + 1):
        x.append([1]) # aspect
        aspect = args.aspects[i - 1]
        node_name.append(aspect)
        weight = 1.0
        edge_index.append([0, i])
        edge_index.append([i, 0])
        edge_weight.append(weight)
        edge_weight.append(weight)

    num_nodes = len(x)
    x = torch.FloatTensor(x)
    edge_index = torch.LongTensor(edge_index).T
    edge_weight = torch.FloatTensor(edge_weight)
    y = torch.LongTensor([num_nodes])

    return Data(
        num_nodes=num_nodes, 
        x=x, 
        y=y, 
        edge_index=edge_index, 
        edge_weight=edge_weight, 
        node_name=node_name
    )

def dataset2graphs(dataset_df: pd.DataFrame, args: Any) -> Tuple[Dict[str, Data], Dict[str, Data]]:
    users = dataset_df["user_id"].unique()
    users_graphs = dict()
    for user_id in tqdm(users, desc="Creating user graphs", colour="green"):
        user_df = dataset_df[dataset_df["user_id"] == user_id]
        graph = document2graph(user_df, user_based=True, args=args)
        users_graphs[user_id] = graph
    
    items = dataset_df["item_id"].unique()
    items_graphs = dict()
    for item_id in tqdm(items, desc="Creating item graphs", colour="green"):
        item_df = dataset_df[dataset_df["item_id"] == item_id]
        graph = document2graph(item_df, user_based=False, args=args)
        items_graphs[item_id] = graph

    return users_graphs, items_graphs


class ABSAGNNRecoDataset(Dataset):

    def __init__(self, users_graphs, items_graphs, data_df, args):
        super().__init__()
        self.users_graphs = users_graphs
        self.items_graphs = items_graphs
        self.data_df = data_df
        self.args = args

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        user_id = row[self.args.user_id_column]
        item_id = row[self.args.item_id_column]
        global_rating = row[self.args.rating_column]
        aspects_ratings = [row[aspect] for aspect in self.args.aspects]
        user_graph = self.users_graphs[user_id]
        item_graph = self.items_graphs[item_id]
        return user_graph, item_graph, aspects_ratings, global_rating


def test(args: Any):
    from torch_geometric.loader import DataLoader

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)
    if args.dataset_path == "":
        seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
        args.dataset_path = os.path.join(seen_dir, "test.csv")
    
    assert args.aspects.strip() != "", "You must specify aspects!"
    args.aspects = args.aspects.strip().split(args.aspects_sep)

    test_df = pd.read_csv(args.dataset_path)

    users_graphs, items_graphs = dataset2graphs(test_df, args)
    dataset = ABSAGNNRecoDataset(users_graphs, items_graphs, test_df, args)
    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))
    print(batch)
    
    user_id = list(users_graphs.keys())[0]
    graph = users_graphs[user_id]

    G = to_networkx(graph)
    G = nx.relabel_nodes(G, dict(zip(range(len(graph.node_name)), graph.node_name)))
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, 
        pos, 
        with_labels=True, 
        node_color=list(map(lambda x: x[0], graph.x)), 
        node_size=500, 
        font_size=10, 
        edge_color="gray"
    )
    plt.title("User Graph")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, default=os.path.join("datasets", "processed"))
    parser.add_argument("--dataset_name", type=str, default="TripAdvisor")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    
    parser.add_argument("--aspects", type=str, 
        default="service cleanliness value sleep_quality rooms location")
    parser.add_argument("--aspects_sep", type=str, default=" ")
    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")

    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    parser.add_argument("--aspect_min_rating", type=float, default=1.0)
    parser.add_argument("--aspect_max_rating", type=float, default=5.0)

    args = parser.parse_args()
    test(args)
