# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# A2R2: Data


import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Any, Tuple

from utils import Vocabulary, create_vocab_from_df


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
    

def get_train_test_data(args: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.dataset_path == "" and (args.train_dataset_path == "" or args.test_dataset_path == ""):
        seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
        args.train_dataset_path = os.path.join(seen_dir, "train.csv")
        args.test_dataset_path = os.path.join(seen_dir, "test.csv")

    if args.train_dataset_path != "" and args.test_dataset_path != "":
        train_df = pd.read_csv(args.train_dataset_path)
        test_df = pd.read_csv(args.test_dataset_path)
        
    else:
        data_df = pd.read_csv(args.dataset_path)
        train_df = data_df.sample(frac=args.train_size, random_state=args.random_state)
        test_df = data_df.drop(train_df.index)

    train_df[args.user_id_column] = train_df[args.user_id_column].apply(str)
    train_df[args.item_id_column] = train_df[args.item_id_column].apply(str)
    test_df[args.user_id_column] = test_df[args.user_id_column].apply(str)
    test_df[args.item_id_column] = test_df[args.item_id_column].apply(str)

    return train_df, test_df


def get_test_data(args: Any) -> pd.DataFrame:
    if args.test_dataset_path == "":
        seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
        args.test_dataset_path = os.path.join(seen_dir, "test.csv")

    test_df = pd.read_csv(args.test_dataset_path)
    test_df[args.user_id_column] = test_df[args.user_id_column].apply(str)
    test_df[args.item_id_column] = test_df[args.item_id_column].apply(str)
    return test_df


def get_vocabularies(args: Any) -> Tuple[Vocabulary, Vocabulary]:
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
        users_vocab.save(os.path.join(metadata_dir, "users_vocab.json"))

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
        items_vocab.save(os.path.join(metadata_dir, "items_vocab.json"))

    return users_vocab, items_vocab
