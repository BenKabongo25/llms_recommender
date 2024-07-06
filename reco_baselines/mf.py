# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Classical RSs
# Matrix factorization

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
from surprise import Dataset, Reader, Trainset
from surprise.prediction_algorithms.matrix_factorization import SVD, SVDpp, NMF
from surprise.model_selection import train_test_split
from typing import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
warnings.filterwarnings(action="ignore")

from common.utils.evaluation import ratings_evaluation


class MFRecommender:

    def __init__(self, args):
        self.args = args

        if self.args.algo == "svd":
            self.algo = SVD(
                n_factors=self.args.n_factors,
                n_epochs=self.args.n_epochs,
                biased=self.args.biased,
                lr_all=self.args.lr_all,
                reg_all=self.args.reg_all,
                random_state=self.args.random_state,
                verbose=self.args.verbose
            )
        elif self.args.algo == "svd++":
            self.algo = SVDpp(
                n_factors=self.args.n_factors,
                n_epochs=self.args.n_epochs,
                lr_all=self.args.lr_all,
                reg_all=self.args.reg_all,
                random_state=self.args.random_state,
                verbose=self.args.verbose
            )
        elif self.args.algo == 'nmf':
            self.algo = NMF(
                n_factors=self.args.n_factors,
                n_epochs=self.args.n_epochs,
                biased=self.args.biased,
                lr_bu=self.args.lr_all,
                lr_bi=self.args.lr_all,
                reg_pu=self.args.reg_all,
                reg_qi=self.args.reg_all,
                reg_bu=self.args.reg_all,
                reg_bi=self.args.reg_all,
                random_state=self.args.random_state,
                verbose=self.args.verbose
            )
        else:
            raise Exception("Unknown MF algo. Accepted values: svd, svd++, nmf.")
        
    def train(self, trainset):
        self.algo.fit(trainset)

    def evaluate(self, testset: Union[List, Trainset]) -> Dict:
        triplets = testset
        if isinstance(testset, Trainset):
            triplets = list(testset.all_ratings())
        
        predictions = []
        references = []
        for (uid, iid, rui) in triplets:
            if isinstance(testset, Trainset):
                uid = testset.to_raw_uid(uid)
                iid = testset.to_raw_iid(iid)
            pui = self.algo.predict(uid, iid).est
            predictions.append(pui)
            references.append(rui)
        scores = ratings_evaluation(predictions, references, self.args)
        return scores


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
    
    train_df = train_df[[args.user_id_column, args.item_id_column, args.rating_column]]
    test_df = test_df[[args.user_id_column, args.item_id_column, args.rating_column]]
    reader = Reader(rating_scale=(args.min_rating, args.max_rating))
    trainset = Dataset.load_from_df(train_df, reader).build_full_trainset()
    testset = Dataset.load_from_df(test_df, reader).build_full_trainset().build_testset()

    model = MFRecommender(args)

    if args.exp_name == "":
        args.exp_name = f"mf_{args.algo}_{args.time_id}"
    exps_base_dir = os.path.join(args.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.log_file_path = os.path.join(exp_dir, "log.txt")
    args.res_file_path = os.path.join(exp_dir, "res.json")

    if args.verbose:
        log = (
            f"Task: Rating prediction\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Algo: {args.algo}\n\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{train_df.head(5)}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    model.train(trainset)
    train_results = model.evaluate(trainset)
    test_results = model.evaluate(testset)

    results = {"train": train_results, "test": test_results}
    if args.verbose:
        print(results)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(str(results))
    with open(args.res_file_path, "w") as res_file:
        json.dump(results, res_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, default="svd++") # svd, svd++, nmf
    parser.add_argument("--n_factors", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--biased", action=argparse.BooleanOptionalAction)
    parser.set_defaults(biased=True)
    parser.add_argument("--lr_all", type=float, default=0.005)
    parser.add_argument("--reg_all", type=float, default=0.02)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)

    parser.add_argument("--base_dir", type=str, default="Datasets\\processed")
    parser.add_argument("--dataset_name", type=str, default="RateBeer")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--train_size", type=float, default=0.8)

    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")
    
    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)

    args = parser.parse_args()
    main(args)
