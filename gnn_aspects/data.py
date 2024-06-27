# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# GNN aspects model: Data

import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from typing import Any, List, Tuple


def document2graph(
    documents_df: pd.DataFrame, 
    user_based: bool=True, 
    args: Any=None
) -> Tuple[Data, List[str]]:
    
    x = []
    node_name = []
    edge_index = []
    edge_weight = []

    x.append([0]) # user(item)
    node_name.append("u" if user_based else "i")

    for i in range(1, len(args.aspects) + 1):
        x.append([1]) # aspect
        node_name.append(args.aspects[i - 1])
        edge_index.append([0, i])
        edge_index.append([i, 0])
        edge_weight.append(1.0)
        edge_weight.append(1.0)

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

    x = torch.FloatTensor(x)
    edge_index = torch.LongTensor(edge_index).T
    edge_weight = torch.FloatTensor(edge_weight)

    return Data(num_nodes=len(x), x=x, edge_index=edge_index, edge_weight=edge_weight, node_name=node_name)


def test(args: Any):
    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)
    if args.dataset_path == "":
        seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
        args.dataset_path = os.path.join(seen_dir, "test.csv")
    
    assert args.aspects.strip() != "", "You must specify aspects!"
    args.aspects = args.aspects.strip().split(args.aspects_sep)

    test_df = pd.read_csv(args.dataset_path)

    item_id = 100542
    document = test_df[test_df["item_id"] == item_id]
    print(document.head(10))
    graph = document2graph(document, user_based=False, args=args)

    G = to_networkx(graph, to_undirected=True)
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
    plt.title("Item Graph")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, default="Datasets\\processed")
    parser.add_argument("--dataset_name", type=str, default="TripAdvisor")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    
    parser.add_argument("--aspects", type=str, 
        default="service cleanliness value sleep_quality rooms location")
    parser.add_argument("--aspects_sep", type=str, default=" ")

    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    parser.add_argument("--aspect_min_rating", type=float, default=1.0)
    parser.add_argument("--aspect_max_rating", type=float, default=5.0)

    args = parser.parse_args()
    test(args)




