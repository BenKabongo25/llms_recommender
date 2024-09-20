# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Classical RSs
# Simple dataset statistics

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import warnings
from typing import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
warnings.filterwarnings(action="ignore")

from common.utils.evaluation import ratings_evaluation


class SimpleStatistics:

    def __init__(self, args):
        self.args = args
        
    def train(self, train_df: pd.DataFrame):
        raise NotImplementedError
    
    def predict(self, uid, iid):
        raise NotImplementedError

    def evaluate(self, test_df: pd.DataFrame) -> Dict:
        predictions = []
        references = []
        for i, row in test_df.iterrows():
            uid = row[self.args.user_id_column]
            iid = row[self.args.item_id_column]
            rui = row[self.args.rating_column]
            pui = self.predict(uid, iid)
            predictions.append(pui)
            references.append(rui)
        scores = ratings_evaluation(predictions, references, self.args)
        return scores
    

class GlobalAvgRating(SimpleStatistics):

    def train(self, train_df: pd.DataFrame):
        self.global_avg = train_df[self.args.rating_column].mean()

    def predict(self, uid, iid):
        return self.global_avg
    

class UserAvgRating(SimpleStatistics):
    
    def train(self, train_df: pd.DataFrame):
        self.user_avg = train_df.groupby(self.args.user_id_column)[self.args.rating_column].mean().to_dict()
        self.global_avg = train_df[self.args.rating_column].mean()
    
    def predict(self, uid, iid):
        return self.user_avg.get(uid, self.global_avg)


class ItemAvgRating(SimpleStatistics):

    def train(self, train_df: pd.DataFrame):
        self.item_avg = train_df.groupby(self.args.item_id_column)[self.args.rating_column].mean().to_dict()
        self.global_avg = train_df[self.args.rating_column].mean()
    
    def predict(self, uid, iid):
        return self.item_avg.get(iid, self.global_avg)
    

def get_method(args):
    if args.method == "global":
        return GlobalAvgRating(args)
    elif args.method == "user":
        return UserAvgRating(args)
    elif args.method == "item":
        return ItemAvgRating(args)
    else:
        raise Exception("Unknown method. Accepted values: global, user, item.")
    

def main(args):
    args.time_id = int(time.time())
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)

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
    
    train_df = train_df[[args.user_id_column, args.item_id_column, args.rating_column]].dropna()
    test_df = test_df[[args.user_id_column, args.item_id_column, args.rating_column]].dropna()

    model = get_method(args)
    if args.exp_name == "":
        args.exp_name = f"{args.method}_{args.algo}_{args.time_id}"
    exps_base_dir = os.path.join(args.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.log_file_path = os.path.join(exp_dir, "log.txt")
    args.res_file_path = os.path.join(exp_dir, "res.json")

    if args.verbose:
        log = (
            f"Task: Rating prediction\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Method: {args.method}\n\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{train_df.head(5)}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    model.train(train_df)
    train_results = model.evaluate(train_df)
    test_results = model.evaluate(test_df)

    results = {"train": train_results, "test": test_results}
    if args.verbose:
        print(results)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(str(results))
    with open(args.res_file_path, "w") as res_file:
        json.dump(results, res_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="item") # global, user, item
    parser.add_argument("--base_dir", type=str, default=os.path.join("datasets", "processed"))
    parser.add_argument("--dataset_name", type=str, default="TripAdvisor")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="global_avg")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")
    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    args = parser.parse_args()
    main(args)
