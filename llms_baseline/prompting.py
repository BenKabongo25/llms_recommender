# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Basline approach
# Prompting - Zero and Few shot

import argparse
import json
import os
import numpy as np
import pandas as pd
import random
import sys
import time
import torch

from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import *
from tqdm import tqdm

from data import DatasetCreator, DataSplitter, TextDataset
from prompters import TargetFormer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from common.utils.evaluation import evaluate_fn


class BasePromptModel:

    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def process_batch(self, batch, device=None):
        raise NotImplementedError
    

class GPT2PromptModel(BasePromptModel):

    def process_batch(self, batch, device=None):
        input_ids = self.tokenizer(
            batch["source_text"], 
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = input_ids.to(device)

        output_ids = self.model.generate(
            input_ids=input_ids.input_ids, 
            max_length=self.args.max_source_length + self.args.max_target_length,
            max_new_tokens=self.args.max_target_length, 
            num_return_sequences=1
        )

        outputs = []
        for i, out in enumerate(output_ids):
            output = self.tokenizer.decode(out, skip_special_tokens=True)
            output = output[len(batch["source_text"][i]):]
            outputs.append(output)        
        return outputs
    

class T5PromptModel(BasePromptModel):

    def process_batch(self, batch, device=None):
        source = self.tokenizer.batch_encode_plus(
            batch["source_text"],
            max_length=self.args.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = source["input_ids"].to(device)
        attention_mask = source["attention_mask"].to(device)

        output_ids = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            do_sample=False,
            max_length=args.max_target_length
        )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return outputs


def prompting(
    prompt_model: BasePromptModel,
    dataloader: DataLoader,
    device: torch.device, 
    args
) -> Dict:
    reviews_predictions = []
    reviews_references = []
    ratings_predictions = []
    ratings_references = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        outputs = prompt_model.process_batch(batch, device)

        if args.verbose and i == 0:
            log = "\n\nExamples:\n"
            for i in range(len(outputs)):
                log += (
                    f"\nInput: {batch['source_text'][i]}\n\n" +
                    f"Target: {batch['target_text'][i]}\n\n" +
                    f"Output: {outputs[i]}\n\n" +
                    f"{'=' * 80}\n\n"
                )
            print(log)
            with open(args.log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(log)

        if args.target_review_flag and args.target_rating_flag:
            outputs_reviews, outputs_ratings = [], []
            for output in outputs:
                review, rating = TargetFormer.get_review_rating(output)
                outputs_reviews.append(review)
                outputs_ratings.append(rating)
            reviews_predictions.extend(outputs_reviews)
            reviews_references.extend(batch["review"])
            ratings_predictions.extend(outputs_ratings)
            ratings_references.extend(batch["rating"])
                
        elif args.target_review_flag:
            reviews_predictions.extend(outputs)
            reviews_references.extend(batch["review"])

        elif args.target_rating_flag:
            ratings_predictions.extend(outputs)
            ratings_references.extend(batch["rating"])

        if args.verbose and i % args.evaluate_every == 0:
            scores = evaluate_fn(reviews_predictions, reviews_references,
                                ratings_predictions, ratings_references, args)
            with open(args.log_file_path, "a") as log_file:
                log_file.write(f"\n\n{args.batch_size * i} samples:\n" + str(scores))

    scores = evaluate_fn(reviews_predictions, reviews_references,
                        ratings_predictions, ratings_references, args)
    if args.verbose:
        with open(args.log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n\nResults: " + str(scores))
    return scores


def get_prompt_model(args):
    if "gpt2" in args.model_name_or_path:
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
        return GPT2PromptModel(model, tokenizer, args)

    if "t5" in args.model_name_or_path:
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        return T5PromptModel(model, tokenizer, args)
    
    return None


def get_text_data(args):
    if args.text_data_path != "":
        text_df = pd.read_csv(args.text_data_path)
    
    else:
        if args.dataset_path == "":
            args.dataset_path = os.path.join(args.dataset_dir, "samples", "splits", "seen", "test.csv")
        data_df = pd.read_csv(args.dataset_path)

        spliter = DataSplitter(args)
        data_split = spliter.split(data_df)
        sampling_df, base_df = data_split["sampling"], data_split["base"]

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

        creator = DatasetCreator(
            sampling_df=sampling_df,
            base_df=base_df,
            users_df=users_df,
            items_df=items_df,
            args=args
        )
        creator.create_dataset()
        text_df = creator.get_text_df()

        if args.save_data_flag:
            if args.save_data_dir == "":
                args.save_data_dir = os.path.join(args.dataset_dir, "samples", str(args.time_id))
                os.makedirs(args.save_data_dir, exist_ok=True)
                args_file_path = os.path.join(args.save_data_dir, "args.json")
                with open(args_file_path, "w") as args_file:
                    json.dump(vars(args), args_file)
            os.makedirs(args.save_data_dir, exist_ok=True)
            creator.save_data(args.save_data_dir)

    return text_df
    
    
def main(args):
    args.time_id = int(time.time())
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    if args.dataset_dir == "":
        args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)

    text_df = get_text_data(args)
    dataset = TextDataset(text_df)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.exp_name == "":
        args.exp_name = (
            f"{args.model_name_or_path}_prompting_" +
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
        if args.target_review_flag and args.target_rating_flag:
            task_name = "Review and rating prediction"
        elif args.target_review_flag:
            task_name = "Review prediction"
        elif args.target_rating_flag:
            task_name = "Rating prediction"
        else:
            task_name = "No task"

        example = next(iter(dataloader))
        log_example = f"Input: {example['source_text'][0]}"
        log_example += f"\n\nTarget: {example['target_text'][0]}"

        log = (
            f"Model: {args.model_name_or_path}\n" +
            f"Tokenizer: {args.tokenizer_name_or_path}\n" +
            f"Task: {task_name}\n" +
            f"Dataset: {args.dataset_name}\n" +
            f"Approach: Prompting - {args.n_samples} shot - {args.n_reviews} reviews \n" +
            f"Sampling method: {args.sampling_method}\n" +
            f"Device: {device}\n\n" +
            f"Data:\n{text_df.head(2)}\n\n"
            f"Input-Output example:\n{log_example}\n\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    prompt_model = get_prompt_model(args)
    if prompt_model is None:
        raise NotImplementedError
    prompt_model.model.to(device)
    prompt_model.model.eval()

    scores = prompting(prompt_model, dataloader, device, args)
    with open(args.res_file_path, "w") as res_file:
        json.dump(scores, res_file)


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
    parser.add_argument("--users_path", type=str, default="")
    parser.add_argument("--items_path", type=str, default="")
    parser.add_argument("--text_data_path", type=str, default="")
    parser.add_argument("--save_data_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(save_data_flag=True)
    parser.add_argument("--save_data_dir", type=str, default="")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    parser.add_argument("--exp_name", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--evaluate_every", type=int, default=10)

    parser.add_argument("--base_data_size", type=float, default=0.25)
    parser.add_argument("--max_base_data_samples", type=int, default=1_000_000)
    parser.add_argument("--split_method", type=int, default=0)
    parser.add_argument("--sampling_method", type=int, default=1)
    parser.add_argument("--similarity_function", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=42)

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

    # Preprocessing
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
