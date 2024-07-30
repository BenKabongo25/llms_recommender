# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Basline approach
# T5 Fine-tuning and evaluation

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from typing import *

from data import TextDataset, get_train_test_data, get_test_data
from prompters import TargetFormer
from utils import (
    empty_cache,
    evaluate_fn,
    ratings_evaluation,
    reviews_evaluation,
    set_seed
)

warnings.filterwarnings(action="ignore")


def get_task_name(args) -> str:
    if args.target_review_flag and args.target_rating_flag:
        task_name = "Review and rating prediction"
    elif args.target_review_flag:
        task_name = "Review prediction"
    elif args.target_rating_flag:
        task_name = "Rating prediction"
    else:
        task_name = "No task"
    return task_name


def set_objective_column_name(args):
    args.objective_column_name = ""
    if args.target_review_flag and args.target_rating_flag:
        args.objective_column_name = args.objective_column_name
    elif args.target_review_flag:
        args.objective_column_name = "review"
    elif args.target_rating_flag:
        args.objective_column_name = "rating"


def collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = [d[key] for d in batch]
    return collated_batch


def get_evaluation_scores(predictions: list, references: list, args: Any) -> dict:
    scores = {}

    if args.target_rating_flag and args.target_review_flag:
        reviews_predictions = []
        reviews_references = []
        ratings_predictions = []
        ratings_references = []

        for i in range(len(predictions)):
            p_review, p_rating = TargetFormer.get_review_rating(predictions[i])
            r_review, r_rating = TargetFormer.get_review_rating(references[i])

            reviews_predictions.append(p_review)
            reviews_references.append(r_review)
            ratings_predictions.append(p_rating)
            ratings_references.append(r_rating)

        scores = evaluate_fn(
            reviews_predictions, reviews_references,
            ratings_predictions, ratings_references,
            args
        )

    elif args.target_rating_flag:
        scores = ratings_evaluation(predictions, references, args)

    elif args.target_review_flag:
        scores = reviews_evaluation(predictions, references, args)

    return scores


class T5Recommender(nn.Module):

    def __init__(self, args: Any=None):
        super().__init__()
        self.args = args

        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))


def evaluate(model, dataloader, args):
    references = []
    predictions = []

    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), "Eval", colour="cyan", total=len(dataloader)):
        empty_cache()

        sources_text = batch["source_text"]
        targets_text = batch[args.objective_column_name]

        inputs = model.tokenizer(
            sources_text, 
            max_length=args.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, do_sample=False, 
            max_length=args.max_target_length
        )
        outputs_text = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        references.extend(targets_text)
        predictions.extend(outputs_text)

        if batch_idx == 0 and args.verbose:
            log = "=" * 150
            for i in range(len(sources_text)):
                log += f"\nInput: {sources_text[i]}"
                log += f"\nTarget: {targets_text[i]}"
                log += f"\nOutput: {outputs_text[i]}"

            print("\n" + log)
            with open(args.log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(log)

    scores = get_evaluation_scores(
        predictions, 
        references,
        args
    )

    return scores


def train(model, optimizer, dataloader, args):
    running_loss = .0
    model.train()

    for batch_idx, batch in tqdm(enumerate(dataloader), "Training", colour="cyan", total=len(dataloader)):
        empty_cache()

        sources_text = batch["source_text"]
        targets_text = batch[args.objective_column_name]

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    running_loss /= len(dataloader)
    return {"loss": running_loss}


def one_epoch_trainer(
    model,  
    optimizer, 
    train_dataloader, 
    test_dataloader,
    args
):
    train_loss_infos = train(model, optimizer, train_dataloader, args)
    train_epoch_infos = evaluate(model, train_dataloader, args)
    test_epoch_infos = evaluate(model, test_dataloader, args)

    if args.target_rating_flag:
        f1_score = test_epoch_infos["f1"]
        if f1_score > args.best_f1_score:
            model.save(args.save_model_path)
            args.best_f1_score = f1_score
    else:
        meteor_score = test_epoch_infos["METEOR"]["meteor"]
        if meteor_score > args.best_meteor_score:
            model.save(args.save_model_path)
            args.best_meteor_score = meteor_score

    return train_loss_infos, train_epoch_infos, test_epoch_infos


def update_infos(train_infos, test_infos, train_loss_infos, train_epoch_infos, test_epoch_infos):
    train_infos["loss"].append(train_loss_infos["loss"])

    for metric in train_epoch_infos:
        if isinstance(train_epoch_infos[metric], dict):
            if metric not in train_infos:
                train_infos[metric] = {}
                test_infos[metric] = {}

            for k in train_epoch_infos[metric]:
                if k not in train_infos[metric]:
                    train_infos[metric][k] = []
                    test_infos[metric][k] = []
                train_infos[metric][k].append(train_epoch_infos[metric][k])
                test_infos[metric][k].append(test_epoch_infos[metric][k])

        else:
            if metric not in train_infos:
                train_infos[metric] = []
                test_infos[metric] = []
            train_infos[metric].append(train_epoch_infos[metric])
            test_infos[metric].append(test_epoch_infos[metric])

    return train_infos, test_infos


def trainer(
    model, 
    optimizer,
    train_dataloader, 
    test_dataloader,
    n_epochs,
    args,
):
    train_infos={"loss": [],}
    test_infos={}

    progress_bar = tqdm(range(1, 1 + n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_loss_infos, train_epoch_infos, test_epoch_infos = one_epoch_trainer(
            model, 
            optimizer,
            train_dataloader, 
            test_dataloader,
            args
        )
        train_infos, test_infos = update_infos(
            train_infos, test_infos,
            train_loss_infos, train_epoch_infos, test_epoch_infos 
        )

        progress_bar.set_description(
            f"Training [{epoch} / {n_epochs}] " +
            f"Loss: train={train_loss_infos['loss']:.4f} "
        )
    
    return train_infos, test_infos


def main_train_test(args):
    train_df, test_df = get_train_test_data(args)
    train_df["rating"] = train_df["rating"].apply(str)
    test_df["rating"] = test_df["rating"].apply(str)

    train_dataset = TextDataset(train_df)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_dataset = TextDataset(test_df)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    set_objective_column_name(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = T5Recommender(args)
    model.to(args.device)
    if args.save_model_path != "":
        model.load(args.save_model_path)

    if args.exp_name == "":
        args.exp_name = (
            f"{args.model_name_or_path}_finetuning_" +
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
        task_name = get_task_name(args)
        example = next(iter(train_dataloader))
        log_example = f"Input: {example['source_text'][0]}"
        log_example += f"\n\nTarget: {example[args.objective_column_name][0]}"
        log = (
            f"Model: {args.model_name_or_path}\n" +
            f"Tokenizer: {args.tokenizer_name_or_path}\n" +
            f"Task: {task_name}\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Device: {device}\n\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{train_df.head(5)}\n\n" +
            f"Input-Output example:\n{log_example}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)
            
    train_infos, test_infos = trainer(model, train_dataloader, test_dataloader, args)
    results = {"train": train_infos, "test": test_infos}
    with open(args.res_file_path, "w") as res_file:
        json.dump(results, res_file)


def main_test(args):
    test_df = get_test_data(args)
    test_df["rating"] = test_df["rating"].apply(str)
    test_dataset = TextDataset(test_df)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    set_objective_column_name(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = T5Recommender(args)
    model.to(args.device)
    if args.save_model_path != "":
        model.load(args.save_model_path)

    if args.exp_name == "":
        args.exp_name = (
            f"eval_{args.model_name_or_path}_finetuning_" +
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
        task_name = get_task_name(args)
        example = next(iter(test_dataloader))
        log_example = f"Input: {example['source_text'][0]}"
        log_example += f"\n\nTarget: {example[args.objective_column_name][0]}"
        log = (
            f"Model: {args.model_name_or_path}\n" +
            f"Tokenizer: {args.tokenizer_name_or_path}\n" +
            f"Task: {task_name}\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Device: {device}\n\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{test_df.head(5)}\n\n" +
            f"Input-Output example:\n{log_example}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    test_results = evaluate(model, test_dataloader, args)
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

    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_model_path", type=str, default="")

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
    parser.set_defaults(timestamp_flag=False)
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

    parser.add_argument("--train_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(train_flag=True)
    parser.add_argument("--save_data_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(save_data_flag=False)
    parser.add_argument("--save_data_dir", type=str, default="")

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
