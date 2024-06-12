# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# P5: https://arxiv.org/abs/2203.13366
# Fine-tuning and evaluation

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys
import time
import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from typing import *

from p5_utils import P5DataCreator, P5Dataset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
warnings.filterwarnings(action="ignore")

from common.utils.evaluation import ratings_evaluation, reviews_evaluation


class P5Model(nn.Module):

    def __init__(self, args: Any=None):
        super().__init__()
        self.args = args

        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

    def forward(self, inputs_ids, attention_mask=None, labels=None):
        return self.model(input_ids=inputs_ids, attention_mask=attention_mask, labels=labels)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))


def train(model, optimizer, dataloader, args):
    references = []
    predictions = []
    running_loss = .0

    model.train()
    for batch in dataloader:
        optimizer.zero_grad()

        sources_text = batch["source"]
        targets_text = batch["target"]

        inputs = model.tokenizer(
            sources_text, 
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)

        targets = model.tokenizer(
            targets_text, 
            max_length=args.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = targets["input_ids"].to(args.device)
        labels[labels == model.tokenizer.pad_token_id] = -100
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        outputs_text = model.tokenizer.batch_decode(outputs.logits, skip_special_tokens=True)

        references.extend(targets_text)
        predictions.extend(outputs_text)

    running_loss /= len(dataloader)
    if args.prompt_type == 2:
        scores = reviews_evaluation(predictions, references, args)
    else:
        scores = ratings_evaluation(predictions, references, args)

    return {"loss": running_loss, **scores}


def test(model, dataloader, args):
    references = []
    predictions = []
    running_loss = .0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            sources_text = batch["source"]
            targets_text = batch["target"]

            inputs = model.tokenizer(
                sources_text, 
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(args.device)
            attention_mask = inputs["attention_mask"].to(args.device)

            targets = model.tokenizer(
                targets_text, 
                max_length=args.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            labels = targets["input_ids"].to(args.device)
            labels[labels == model.tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()

            outputs_text = model.tokenizer.batch_decode(outputs.logits, skip_special_tokens=True)

            references.extend(targets_text)
            predictions.extend(outputs_text)

    running_loss /= len(dataloader)
    if args.prompt_type == 2:
        scores = reviews_evaluation(predictions, references, args)
    else:
        scores = ratings_evaluation(predictions, references, args)

    return {"loss": running_loss, **scores}


def trainer(model, train_dataloader, test_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_infos = {}
    test_infos = {}

    progress_bar = tqdm(range(1, 1 + args.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_epoch_infos = train(model, optimizer, train_dataloader, args)
        test_epoch_infos = test(model, test_dataloader, args)

        for metric in train_epoch_infos:
            if metric not in train_infos:
                train_infos[metric] = []
                test_infos[metric] = []
            train_infos[metric].append(train_epoch_infos[metric])
            test_infos[metric].append(test_epoch_infos[metric])

        progress_bar.set_description(
            f"[{epoch} / {args.n_epochs}] " +
            f"RMSE: train={train_epoch_infos['RMSE']:.4f} test={test_epoch_infos['RMSE']:.4f} " +
            f"Loss: train={train_epoch_infos['loss']:.4f} test={test_epoch_infos['loss']:.4f}"
        )

        results = {"train": train_infos, "test": test_infos}
        with open(args.res_file_path, "w") as res_file:
            json.dump(results, res_file)

        if epoch % args.save_every == 0:
            model.save(args.save_model_path)

    return train_infos, test_infos


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

        p5_data_creator = P5DataCreator(args)
        train_df = p5_data_creator.create_dataset(
            train_data_df,
            users_df=users_df,
            items_df=items_df
        )
        test_df = p5_data_creator.create_dataset(
            test_data_df,
            users_df=users_df,
            items_df=items_df
        )

        if args.save_data_flag:
            if args.save_data_dir == "":
                args.save_data_dir = os.path.join(args.dataset_dir, "samples", str(args.time_id))            
            os.makedirs(args.save_data_dir, exist_ok=True)
            train_df.to_csv(os.path.join(args.save_data_dir, "train.csv"), index=False)
            test_df.to_csv(os.path.join(args.save_data_dir, "test.csv"), index=False)

    return train_df, test_df


def get_task_name(args: Any) -> str:
    if args.prompt_type == 2:
        task_name = "Review prediction"
    else:
        task_name = "Rating prediction"
    return task_name


def main_train_test(args):
    train_df, test_df = get_train_test_data(args)
    train_dataset = P5Dataset(train_df, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = P5Dataset(test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = P5Model(args)
    model.to(args.device)
    if args.save_model_path != "":
        model.load(args.save_model_path)

    if args.exp_name == "":
        args.exp_name = f"p5_{args.model_name_or_path}_{args.time_id}"
    
    args.exp_name = args.exp_name.replace(" ", "_").replace("/", "_")
    exps_base_dir = os.path.join(args.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.log_file_path = os.path.join(exp_dir, "log.txt")
    args.res_file_path = os.path.join(exp_dir, "res.json")

    if args.save_model_path == "":
        args.save_model_path = os.path.join(exp_dir, "model.pth")

    if args.verbose:
        log = (
            f"Model: {args.model_name_or_path}\n" +
            f"Tokenizer: {args.tokenizer_name_or_path}\n" +
            f"Task: P5 {get_task_name(args)}\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Device: {device}\n\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{train_df.head(5)}\n\n"
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
        plt.title(f"{args.dataset_name} MLP - {metric}")
        plt.plot(train_infos[metric], label="train")
        plt.plot(test_infos[metric], label="test")
        plt.legend()
        plt.savefig(os.path.join(exp_dir, metric.lower() + ".png"))


def get_test_data(args: Any) -> pd.DataFrame:
    if args.test_text_data_path != "":
        test_df = pd.read_csv(args.test_text_data_path)
        return test_df

    else:    
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

        p5_data_creator = P5DataCreator(args)
        test_df = p5_data_creator.create_dataset(
            test_data_df,
            users_df=users_df,
            items_df=items_df
        )

        if args.save_data_flag:
            if args.save_data_dir == "":
                args.save_data_dir = os.path.join(args.dataset_dir, "samples", str(args.time_id))            
            os.makedirs(args.save_data_dir, exist_ok=True)
            test_df.to_csv(os.path.join(args.save_data_dir, "test.csv"), index=False)

    return test_df


def main_test(args):
    test_df = get_test_data(args)
    test_dataset = P5Dataset(test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = P5Model(args)
    model.to(args.device)
    if args.save_model_path != "":
        model.load(args.save_model_path)

    if args.exp_name == "":
        args.exp_name = f"p5_eval_{args.model_name_or_path}_{args.time_id}"
    
    args.exp_name = args.exp_name.replace(" ", "_").replace("/", "_")
    exps_base_dir = os.path.join(args.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.log_file_path = os.path.join(exp_dir, "log.txt")
    args.res_file_path = os.path.join(exp_dir, "res.json")

    if args.verbose:
        log = (
            f"Model: {args.model_name_or_path}\n" +
            f"Tokenizer: {args.tokenizer_name_or_path}\n" +
            f"Task: P5 {get_task_name(args)}\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Device: {device}\n\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{test.head(5)}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    test_results = test(model, test_dataloader, args)
    results = {"train": {}, "test": test_results}
    with open(args.res_file_path, "w") as res_file:
        json.dump(results, res_file)

    log = "Test Results:\n"
    for metric, value in test_results.items():
        log += f"{metric}: {value}\n"
    print(log)
    with open(args.log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(log)


def main(args):
    args.time_id = int(time.time())
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)

    if args.train_flag:
        main_train_test(args)
    else:
        main_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="google/flan-t5-base")
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=128)

    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--users_path", type=str, default="")
    parser.add_argument("--items_path", type=str, default="")
    parser.add_argument("--train_text_data_path", type=str, default="")
    parser.add_argument("--test_text_data_path", type=str, default="")

    parser.add_argument("--train_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(train_flag=True)
    parser.add_argument("--save_data_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(save_data_flag=False)
    parser.add_argument("--save_data_dir", type=str, default="")

    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="")

    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_model_path", type=str, default="")

    parser.add_argument("--prompt_type", type=int, default=0)
    parser.add_argument("--max_review_length", type=int, default=128)
    parser.add_argument('--max_description_length', type=int, default=128)
    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")
    parser.add_argument("--review_column", type=str, default="review")

    parser.add_argument("--user_description_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(user_description_flag=False)
    parser.add_argument("--item_description_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(item_description_flag=True)
    parser.add_argument("--user_description_column", type=str, default="description")
    parser.add_argument("--item_description_column", type=str, default="description")

    parser.add_argument("--truncate_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(truncate_flag=True)
    parser.add_argument("--lower_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lower_flag=True)
    parser.add_argument("--delete_balise_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_balise_flag=True)
    parser.add_argument("--delete_stopwords_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_stopwords_flag=False)
    parser.add_argument("--delete_punctuation_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_punctuation_flag=False)
    parser.add_argument("--delete_non_ascii_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_non_ascii_flag=True)
    parser.add_argument("--delete_digit_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_digit_flag=False)
    parser.add_argument("--replace_maj_word_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(replace_maj_word_flag=False)
    parser.add_argument("--first_line_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(first_line_flag=False)
    parser.add_argument("--last_line_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(last_line_flag=False)
    parser.add_argument("--stem_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(stem_flag=False)
    parser.add_argument("--lemmatize_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lemmatize_flag=False)

    args = parser.parse_args()
    main(args)
