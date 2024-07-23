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

from annotations_text import AnnotationsTextFormerBase
from data import T5ABSADataset, collate_fn, get_train_val_test_df
from enums import TaskType, AbsaTupleType, AnnotationsTextFormerType
from eval import get_evaluation_scores
from utils import AbsaData, set_seed


def save_model(model, path: str):
    torch.save(model.state_dict(), path)


def load_model(model, path: str):
    model.load_state_dict(torch.load(path))


def train_test(
    model, 
    tokenizer, 
    annotations_text_former, 
    optimizer, 
    dataloader, 
    args,
    train_flag=True, 
    epoch=-1
):
    references = []
    predictions = []
    all_annotations = []

    running_loss = .0
    if train_flag: model.train()
    else: model.eval()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)
        
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

        target_texts = batch["target_texts"]
        annotations = batch["annotations"]

        references.extend(target_texts)
        predictions.extend(output_texts)
        all_annotations.extend(annotations)

        if batch_idx == 0 and args.verbose and epoch % args.verbose_every == 0:
            input_texts = batch["input_texts"]
            log = "=" * 100
            is_train = "Train" if train_flag else "Eval"
            log += f"\n{is_train}: Epoch {epoch}/{args.n_epochs}"
            for i in range(len(input_texts)):
                log += f"\nInput: {input_texts[i]}"
                log += f"\nTarget: {target_texts[i]}"
                log += f"\nAnnotations: {annotations[i]}"
                log += f"\nOutput: {output_texts[i]}"
                if args.task_type is TaskType.T2A:
                    log += (
                        f"\nOutput annotations: "
                        f"{annotations_text_former.multiple_text_to_annotations(output_texts[i])}"
                    )
                log += "\n"
                
            print("\n" + log)
            with open(args.log_file_path, "w", encoding="utf-8") as log_file:
                log_file.write(log)

    running_loss /= len(dataloader)
    scores = get_evaluation_scores(
        predictions, 
        references, 
        all_annotations, 
        annotations_text_former,
        args
    )

    return {"loss": running_loss, **scores}


def trainer(
    model, 
    tokenizer, 
    annotations_text_former, 
    train_dataloader, 
    test_dataloader, 
    args
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_infos = {}
    test_infos = {}

    progress_bar = tqdm(range(1, 1 + args.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        train_epoch_infos = train_test(
            model, 
            tokenizer, 
            annotations_text_former, 
            optimizer, 
            train_dataloader, 
            args, 
            True, 
            epoch
        )

        with torch.no_grad():
            test_epoch_infos = train_test(
                model, 
                tokenizer, 
                annotations_text_former, 
                None, 
                test_dataloader, 
                args, 
                False, 
                epoch
            )

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

    args.task_type = TaskType(args.task_type)
    args.absa_tuple = AbsaTupleType(args.absa_tuple)
    args.annotations_text_type = AnnotationsTextFormerType(args.annotations_text_type)

    absa_data = AbsaData()
    annotations_text_former = AnnotationsTextFormerBase.get_annotations_text_former(args, absa_data)

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

    train_df, val_df, test_df = get_train_val_test_df(args)
    
    train_dataset = T5ABSADataset(tokenizer, annotations_text_former, train_df, args)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_dataset = T5ABSADataset(tokenizer, annotations_text_former, val_df, args)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    test_dataset = T5ABSADataset(tokenizer, annotations_text_former, test_df, args)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    if args.save_model_path != "":
        load_model(model, args.save_model_path)

    if args.exp_name == "":
        args.exp_name = (
            f"{args.model_name_or_path}_{args.task_type.value}_{args.absa_tuple.value}_"
            f"{args.annotations_text_type.value}_{args.time_id}"
        )
    
    args.exp_name = args.exp_name.replace(" ", "_").replace("/", "_")
    exps_base_dir = os.path.join(args.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.exp_dir = exp_dir
    args.log_file_path = os.path.join(exp_dir, "log.txt")
    args.res_file_path = os.path.join(exp_dir, "res.json")

    if args.save_model_path == "":
        args.save_model_path = os.path.join(exp_dir, "model.pth")

    if args.verbose:
        batch = next(iter(train_dataloader))
        input_texts = batch["input_texts"]
        target_texts = batch["target_texts"]
        annotations = batch["annotations"]
        input_ids = batch["input_ids"]

        log_example = f"Input: {input_texts[0]}"
        log_example += f"\nTarget: {target_texts[0]}"
        log_example += f"\nAnnotations: {annotations[0]}"
        log_example += f"\nInputs size: {input_ids.shape}"
        log = (
            f"Model: {args.model_name_or_path}\n" +
            f"Tokenizer: {args.tokenizer_name_or_path}\n" +
            f"Task: {args.task_type}\n" +
            f"Tuple: {args.absa_tuple}\n" +
            f"Annotations: {args.annotations_text_type}\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Device: {device}\n" +
            f"Arguments:\n{args}\n" +
            f"Data:\n{train_df.head(5)}\n" +
            f"Input-Output example:\n{log_example}\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    train_infos, val_infos = trainer(
        model, 
        tokenizer, 
        annotations_text_former, 
        train_dataloader, 
        val_dataloader, 
        args
    )

    with torch.no_grad():
        test_infos = train_test(
            model, 
            tokenizer, 
            annotations_text_former, 
            None, 
            test_dataloader, 
            args, 
            train_flag=False
        )
    
    results = {"test": test_infos, "train": train_infos, "val": val_infos}
    with open(args.res_file_path, "w") as res_file:
        json.dump(results, res_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="t5-base")
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)

    parser.add_argument("--task_type", type=str, default=TaskType.T2A.value)
    parser.add_argument("--absa_tuple", type=str, default=AbsaTupleType.ACOP.value)
    parser.add_argument("--annotations_text_type", type=str, 
        default=AnnotationsTextFormerType.GAS_EXTRACTION_STYLE.value)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--annotations_column", type=str, default="annotations")
    parser.add_argument("--prompt_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(prompt_flag=False)

    parser.add_argument("--base_dir", type=str, default=os.path.join("datasets", "absa"))
    parser.add_argument("--dataset_name", type=str, default="Rest16")
    parser.add_argument("--dataset_dir", type=str, default=os.path.join("datasets", "absa", "Rest16"))
    parser.add_argument("--dataset_path", type=str, default=os.path.join("datasets", "absa", "Rest16", "data.csv"))
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--val_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)

    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    parser.add_argument("--verbose_every", type=int, default=1)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--save_model_path", type=str, default="")

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
