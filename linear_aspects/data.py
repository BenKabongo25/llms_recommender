# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Linear aspects model: Data


import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Any, Tuple


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

    return train_df, test_df


def get_test_data(args: Any) -> pd.DataFrame:
    if args.test_dataset_path == "":
        seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
        args.test_dataset_path = os.path.join(seen_dir, "test.csv")

    test_df = pd.read_csv(args.test_dataset_path)
    return test_df
