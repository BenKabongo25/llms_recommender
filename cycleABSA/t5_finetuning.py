# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: T5 Fine-tuning

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from typing import *

from data import T5ABSADataset, get_train_val_test_df
from enums import Task, AbsaTuple
from eval import get_evaluation_scores
from utils import set_seed


def save_model(model, path: str):
    torch.save(model.state_dict(), path)


def load_model(model, path: str):
    model.load_state_dict(torch.load(path))


def train_test(model, tokenizer, optimizer, dataloader, args, train_flag=True):
    references = []
    predictions = []
    all_annotations = []

    running_loss = .0
    if train_flag: model.train()
    else: model.eval()

    for batch in dataloader:
        _, target_texts, input_ids, attention_mask, labels, annotations = batch
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        if train_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        output_texts = tokenizer.batch_decode(
            outputs.logits.argmax(dim=-1), 
            skip_special_tokens=True
        )

        references.extend(target_texts)
        predictions.extend(output_texts)
        all_annotations.extend(annotations)

    running_loss /= len(dataloader)
    scores = get_evaluation_scores(predictions, references, all_annotations, args)

    return {"loss": running_loss, **scores}


def trainer(model, tokenizer, train_dataloader, test_dataloader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_infos = {}
    test_infos = {}

    progress_bar = tqdm(range(1, 1 + args.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_epoch_infos = train_test(model, tokenizer, optimizer, train_dataloader, args, train_flag=True)
        with torch.no_grad():
            test_epoch_infos = train_test(model, tokenizer, None, test_dataloader, args, train_flag=False)

        for metric in train_epoch_infos:
            if metric not in train_infos:
                train_infos[metric] = []
                test_infos[metric] = []
            train_infos[metric].append(train_epoch_infos[metric])
            test_infos[metric].append(test_epoch_infos[metric])

        progress_bar.set_description(
            f"[{epoch} / {args.n_epochs}] " +
            f"Loss: train={train_epoch_infos['loss']:.4f} test={test_epoch_infos['loss']:.4f}"
        )

        results = {"train": train_infos, "test": test_infos}
        with open(args.res_file_path, "w") as res_file:
            json.dump(results, res_file)

        if epoch % args.save_every == 0:
            save_model(model, args.save_model_path)

    return train_infos, test_infos


def main(args):
    set_seed(args)

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)

    train_df, val_df, test_df = get_train_val_test_df(args)
    
    train_dataset = T5ABSADataset(train_df)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True
    )

    val_dataset = T5ABSADataset(val_df)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, shuffle=False
    )

    test_dataset = T5ABSADataset(test_df)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

    model.to(args.device)
    if args.save_model_path != "":
        load_model(model, args.save_model_path)

    if args.exp_name == "":
        args.exp_name = (
            f"{args.model_name_or_path}_{args.time_id}"
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
        example = next(iter(train_dataloader))
        log_example = f"Input: {example['source_text'][0]}"
        log_example += f"\n\nTarget: {example[args.objective_column_name][0]}"
        log = (
            f"Model: {args.model_name_or_path}\n" +
            f"Tokenizer: {args.tokenizer_name_or_path}\n" +
            f"Task: {args.task_name}\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Device: {device}\n\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{train_df.head(5)}\n\n" +
            f"Input-Output example:\n{log_example}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    train_infos, val_infos = trainer(model, tokenizer, train_dataloader, val_dataloader, args)
    with torch.no_grad():
        test_infos = train_test(model, tokenizer, None, test_dataloader, args, train_flag=False)
    results = {"test": test_infos, "train": train_infos, "val": val_infos}
    with open(args.res_file_path, "w") as res_file:
        json.dump(results, res_file)


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
    parser.add_argument("--val_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)

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

    parser.add_argument("--train_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(train_flag=True)
    parser.add_argument("--save_data_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(save_data_flag=False)
    parser.add_argument("--save_data_dir", type=str, default="")

    parser.add_argument("--task_name", type=str, default=Task.T2A)
    parser.add_argument("--absa_tuple", type=str, default=AbsaTuple.ACOP)
    parser.add_argument("--input_column", type=str, default="text")
    parser.add_argument("--output_column", type=str, default="aspects")
    parser.add_argument("--annotations_column", type=str, default="annotations")

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