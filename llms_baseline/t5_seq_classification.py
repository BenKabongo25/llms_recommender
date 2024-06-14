# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# T5 for Sequence Classification for Rating prediction task

import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
import torch
import torch.nn as nn
import warnings

from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForSequenceClassification, T5Tokenizer
from tqdm import tqdm
from typing import *

from utils import get_train_test_data, get_test_data

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
warnings.filterwarnings(action="ignore")

from common.utils.evaluation import ratings_evaluation
from common.utils.functions import set_seed


class TextRatingDataset(Dataset):

    def __init__(self, data_df: pd.DataFrame, args: Any=None):
        self.data_df = data_df
        self.args = args

    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, index: int) -> Tuple[str, int]:
        row = self.data_df.iloc[index]
        source = row["source"]
        rating = row["rating"]
        return source, rating


class T5MLPRecommender(nn.Module):

    def __init__(self, n_classes: int=1, args: Any=None):
        super().__init__()
        self.n_classes = n_classes
        self.args = args

        self.model = T5ForSequenceClassification.from_pretrained(args.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

        if args.mlp_classifier_flag:
            self.model.classification_head = nn.Sequential(
                nn.Linear(self.model.config.d_model, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, n_classes)
            )
        else:
            self.model.classification_head = nn.Linear(self.model.config.d_model, n_classes)

        #for param in self.parameters():
        #    param.requires_grad = False
        #self.model.classification_head.requires_grad = True

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))


def get_loss_fn(args: Any) -> nn.Module:
    if args.do_classification:
        return nn.CrossEntropyLoss()
    return nn.MSELoss()


def train(model, optimizer, dataloader, loss_fn, args):
    references = []
    predictions = []
    running_loss = .0

    model.train()
    for source, R in dataloader:
        optimizer.zero_grad()

        inputs = model.tokenizer(
            source, 
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        
        if args.do_classification:
            R = torch.LongTensor(R).to(args.device)
        else:
            R = torch.tensor(R.clone().detach(), dtype=torch.float32).to(args.device)
        
        R_hat = model(input_ids, attention_mask).squeeze()
        loss = loss_fn(R_hat, R)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if args.do_classification:
            R_hat = R_hat.argmax(dim=1)

        references.extend(R.cpu().detach().tolist())
        predictions.extend(R_hat.cpu().detach().tolist())

    running_loss /= len(dataloader)
    ratings_scores = ratings_evaluation(predictions, references, args)
    accuracy = -1
    if args.do_classification:
        accuracy = metrics.accuracy_score(references, predictions)

    return {
        "accuracy": accuracy, 
        "loss": running_loss, 
        "RMSE": ratings_scores["rmse"], 
        "MAE": ratings_scores["mae"], 
        "P": ratings_scores["precision"], 
        "R": ratings_scores["recall"], 
        "F1": ratings_scores["f1"], 
        "AUC": ratings_scores["auc"]
    }


def test(model, dataloader, loss_fn, args):
    references = []
    predictions = []
    running_loss = .0

    model.eval()
    with torch.no_grad():
        for source, R in dataloader:        
            inputs = model.tokenizer(
                source, 
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(args.device)
            attention_mask = inputs["attention_mask"].to(args.device)
            
            if args.do_classification:
                R = torch.LongTensor(R).to(args.device)
            else:
                R = torch.tensor(R.clone().detach(), dtype=torch.float32).to(args.device)
            
            R_hat = model(input_ids, attention_mask).squeeze()
            loss = loss_fn(R_hat, R)
            running_loss += loss.item()

            if args.do_classification:
                R_hat = R_hat.argmax(dim=1)
                
            references.extend(R.cpu().detach().tolist())
            predictions.extend(R_hat.cpu().detach().tolist())

    running_loss /= len(dataloader)
    ratings_scores = ratings_evaluation(predictions, references, args)
    accuracy = -1
    if args.do_classification:
        accuracy = metrics.accuracy_score(references, predictions)

    return {
        "accuracy": accuracy, 
        "loss": running_loss, 
        "RMSE": ratings_scores["rmse"], 
        "MAE": ratings_scores["mae"], 
        "P": ratings_scores["precision"], 
        "R": ratings_scores["recall"], 
        "F1": ratings_scores["f1"], 
        "AUC": ratings_scores["auc"]
    }


def trainer(model, train_dataloader, test_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = get_loss_fn(args)

    train_infos = {
        "accuracy": [], "loss": [], "RMSE": [], "MAE": [], "P": [], "R": [], "F1": [], "AUC": []
    }
    test_infos = {
        "accuracy": [], "loss": [], "RMSE": [], "MAE": [], "P": [], "R": [], "F1": [], "AUC": []
    }

    progress_bar = tqdm(range(1, 1 + args.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_epoch_infos = train(model, optimizer, train_dataloader, loss_fn, args)
        test_epoch_infos = test(model, test_dataloader, loss_fn, args)

        for metric in train_infos:
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


def main_train_test(args):
    train_df, test_df = get_train_test_data(args)

    if args.do_classification:
        rating_fn = lambda x: int(x - args.min_rating)
        train_df["rating"] = train_df["rating"].apply(rating_fn)
        test_df["rating"] = test_df["rating"].apply(rating_fn)

    train_dataset = TextRatingDataset(train_df, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = TextRatingDataset(test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    n_classes = 1
    if args.do_classification:
        n_classes = int(args.max_rating - args.min_rating + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = T5MLPRecommender(n_classes, args)
    model.to(args.device)
    if args.save_model_path != "":
        model.load(args.save_model_path)

    mlp_flag = "mlp" if args.mlp_classifier_flag else "linear"
    if args.exp_name == "":
        args.exp_name = (
            f"{args.model_name_or_path}_{mlp_flag}_" +
            f"{args.n_samples}_shot_{args.n_reviews}_reviews_"
            f"{args.sampling_method}_sampling_{args.time_id}"
        )
    
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
            f"Task: Rating prediction\n" +
            f"MLP Classifier: {args.mlp_classifier_flag}\n" +
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


def main_test(args):
    test_df = get_test_data(args)

    if args.do_classification:
        rating_fn = lambda x: int(x - args.min_rating)
        test_df["rating"] = test_df["rating"].apply(rating_fn)

    test_dataset = TextRatingDataset(test_df, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    n_classes = 1
    if args.do_classification:
        n_classes = int(args.max_rating - args.min_rating + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = T5MLPRecommender(n_classes, args)
    model.to(args.device)
    if args.save_model_path != "":
        model.load(args.save_model_path)

    if args.exp_name == "":
        args.exp_name = (
            f"eval_{args.model_name_or_path}_mlp_" +
            f"{args.n_samples}_shot_{args.n_reviews}_reviews_"
            f"{args.sampling_method}_sampling_{args.time_id}"
        )
    
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
            f"Task: Rating prediction\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Device: {device}\n\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{test.head(5)}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    loss_fn = get_loss_fn(args)
    test_results = test(model, test_dataloader, loss_fn, args)
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
    set_seed(args)

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

    parser.add_argument("--do_classification", action=argparse.BooleanOptionalAction)
    parser.set_defaults(do_classification=False)
    
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_model_path", type=str, default="")
    parser.add_argument('--mlp_classifier_flag', action=argparse.BooleanOptionalAction)
    parser.set_defaults(mlp_classifier_flag=True)

    parser.add_argument("--base_data_size", type=float, default=0.25)
    parser.add_argument("--max_base_data_samples", type=int, default=1_000_000)
    parser.add_argument("--split_method", type=int, default=0)
    parser.add_argument("--sampling_method", type=int, default=1)
    parser.add_argument("--similarity_function", type=int, default=0)

    parser.add_argument("--n_reviews", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=0)
    parser.add_argument("--max_review_length", type=int, default=128)
    parser.add_argument('--max_description_length', type=int, default=128)
    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)
    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="item_id")
    parser.add_argument("--rating_column", type=str, default="rating")
    parser.add_argument("--review_column", type=str, default="review")
    parser.add_argument("--timestamp_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(timestamp_flag=True)
    parser.add_argument("--timestamp_column", type=str, default="timestamp")

    parser.add_argument("--user_description_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(user_description_flag=False)
    parser.add_argument("--item_description_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(item_description_flag=True)
    parser.add_argument("--user_only_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(user_only_flag=False)
    parser.add_argument("--user_description_column", type=str, default="description")
    parser.add_argument("--item_description_column", type=str, default="description")
    parser.add_argument("--source_review_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(source_review_flag=True)
    parser.add_argument("--source_rating_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(source_rating_flag=False)
    parser.add_argument("--user_first_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(user_first_flag=True)
    parser.add_argument("--target_review_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(target_review_flag=False)
    parser.add_argument("--target_rating_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(target_rating_flag=True)

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
