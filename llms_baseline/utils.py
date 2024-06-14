# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Basline approach
# Utils

import json
import os
import pandas as pd
from typing import *

from data import DataSplitter, DatasetCreator


def get_train_test_data(args: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.train_text_data_path != "" and args.test_text_data_path != "":
        train_df = pd.read_csv(args.train_text_data_path)
        test_df = pd.read_csv(args.test_text_data_path)
    
    else:
        if args.dataset_path == "" and (args.train_dataset_path == "" or args.test_dataset_path == ""):
            seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
            args.train_dataset_path = os.path.join(seen_dir, "train.csv")
            args.test_dataset_path = os.path.join(seen_dir, "test.csv")

        if args.train_dataset_path != "" and args.test_dataset_path != "":
            train_data_df = pd.read_csv(args.train_dataset_path)
            test_data_df = pd.read_csv(args.test_dataset_path)
        
        else:
            data_df = pd.read_csv(args.dataset_path)
            train_data_df = data_df.sample(frac=args.train_size, random_state=args.random_state)
            test_data_df = data_df.drop(train_data_df.index)

        metadata_dir = os.path.join(args.dataset_dir, "samples", "metadata")
        users_df = None
        if args.user_description_flag:
            if args.users_path == "":
                args.users_path = os.path.join(metadata_dir, "users.csv")
            users_df = pd.read_csv(args.users_path)

        items_df = None
        if args.item_description_flag:
            if args.items_path == "":
                args.items_path = os.path.join(metadata_dir, "items.csv")
            items_df = pd.read_csv(args.items_path)

        spliter = DataSplitter(args)
        train_split = spliter.split(train_data_df)
        test_split = spliter.split(test_data_df)

        train_creator = DatasetCreator(
            sampling_df=train_split["sampling"],
            base_df=train_split["base"],
            users_df=users_df,
            items_df=items_df,
            args=args
        )
        train_creator.create_dataset()
        train_df = train_creator.get_text_df()

        test_creator = DatasetCreator(      
            sampling_df=test_split["sampling"],
            base_df=test_split["base"],
            users_df=users_df,
            items_df=items_df,
            args=args
        )
        test_creator.create_dataset()
        test_df = test_creator.get_text_df()

        if args.save_data_flag:
            if args.save_data_dir == "":
                args.save_data_dir = os.path.join(args.dataset_dir, "samples", str(args.time_id))
                os.makedirs(args.save_data_dir, exist_ok=True)
                args_file_path = os.path.join(args.save_data_dir, "args.json")
                with open(args_file_path, "w") as args_file:
                    json.dump(vars(args), args_file)

            train_save_dir = os.path.join(args.save_data_dir, "train")
            test_save_dir = os.path.join(args.save_data_dir, "test")
            
            os.makedirs(args.save_data_dir, exist_ok=True)
            os.makedirs(train_save_dir, exist_ok=True)
            os.makedirs(test_save_dir, exist_ok=True)

            train_creator.save_data(train_save_dir)
            test_creator.save_data(test_save_dir)

    return train_df, test_df


def get_test_data(args: Any) -> pd.DataFrame:
    if args.test_text_data_path != "":
        test_df = pd.read_csv(args.test_text_data_path)
        return test_df

    if args.test_dataset_path == "":
        seen_dir = os.path.join(args.dataset_dir, "samples", "splits", "seen")
        args.test_dataset_path = os.path.join(seen_dir, "test.csv")

    test_data_df = pd.read_csv(args.test_dataset_path)
            
    metadata_dir = os.path.join(args.dataset_dir, "samples", "metadata")
    users_df = None
    if args.user_description_flag:
        if args.users_path == "":
            args.users_path = os.path.join(metadata_dir, "users.csv")
        users_df = pd.read_csv(args.users_path)

    items_df = None
    if args.item_description_flag:
        if args.items_path == "":
            args.items_path = os.path.join(metadata_dir, "items.csv")
        items_df = pd.read_csv(args.items_path)

    spliter = DataSplitter(args)
    test_split = spliter.split(test_data_df)

    test_creator = DatasetCreator(      
        sampling_df=test_split["sampling"],
        base_df=test_split["base"],
        users_df=users_df,
        items_df=items_df,
        args=args
    )
    test_creator.create_dataset()
    test_df = test_creator.get_text_df()

    if args.save_data_flag:
        if args.save_data_dir == "":
            args.save_data_dir = os.path.join(args.dataset_dir, "samples", str(args.time_id))
            os.makedirs(args.save_data_dir, exist_ok=True)
            args_file_path = os.path.join(args.save_data_dir, "args.json")
            with open(args_file_path, "w") as args_file:
                json.dump(vars(args), args_file)

        test_save_dir = os.path.join(args.save_data_dir, "test")
        os.makedirs(args.save_data_dir, exist_ok=True)
        os.makedirs(test_save_dir, exist_ok=True)
        test_creator.save_data(test_save_dir)

    return test_df