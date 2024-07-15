# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: Data

import ast
import pandas as pd
import pytorch_lightning as pl
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5Tokenizer
from typing import *

from enums import Task
from prompts import get_prompt
from utils import preprocess_text


class T5ABSADataset(Dataset):

    def __init__(
        self,
        tokenizer: T5Tokenizer, 
        data_df: pd.DataFrame,
        args: Any
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_texts = data_df[args.input_column].tolist()
        self.target_texts = data_df[args.target_column].tolist()
        self.aspects_annotations = data_df[args.annotations_column].apply(ast.literal_eval).tolist()
        self.args = args

        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        self._build()

    def __len__(self):
        return len(self.input_texts)
    
    def _build(self):
        for idx in tqdm(range(len(self.input_texts)), "Prepare data", colour="green"):
            input_text = self.input_texts[idx]
            target_text = self.target_texts[idx]
            annotations = self.aspects_annotations[idx]

            if self.args.task_name is Task.T2A:
                input_text = preprocess_text(input_text, self.args, self.args.max_input_length)
            else:
                target_text = preprocess_text(target_text, self.args, self.args.max_target_length)

            input_text = get_prompt(text=input_text, annotations=annotations, args=self.args)
            self.input_texts[idx] = input_text
            self.target_texts[idx] = target_text
            
            input = self.tokenizer(
                input_text, 
                max_length=self.args.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = input["input_ids"]
            attention_mask = input["attention_mask"]

            target = self.tokenizer(
                target_text, 
                max_length=self.args.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            labels = target["input_ids"]
            labels[labels == self.tokenizer.pad_token_id] = -100

            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.labels.append(labels)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        labels = self.labels[idx]
        annotations = self.aspects_annotations[idx]
        #return input_text, target_text, input_ids, attention_mask, labels, annotations
        return {
            "input_texts": input_text,
            "target_texts": target_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "annotations": annotations
        }
        

def get_train_val_test_df(args: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.train_dataset_path != "" and args.test_dataset_path != "":
        train_df = pd.read_csv(args.train_dataset_path)
        test_df = pd.read_csv(args.test_dataset_path)
        if args.val_dataset_path is not None:
            val_df = pd.read_csv(args.val_dataset_path)
        else:
            train_df, val_df = train_test_split(
                train_df, 
                test_size=args.val_size, 
                random_state=args.random_state
            )
    else:
        data_df = pd.read_csv(args.dataset_path)
        train_df, test_df = train_test_split(
            data_df, 
            test_size=args.test_size, 
            random_state=args.random_state
        )
        train_df, val_df = train_test_split(
            train_df, 
            test_size=args.val_size, 
            random_state=args.random_state
        )
    return train_df, val_df, test_df


def collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = [d[key] for d in batch]
        if isinstance(collated_batch[key][0], torch.Tensor):
            collated_batch[key] = torch.cat(collated_batch[key], 0)
    return collated_batch


class T5DataModule(pl.LightningDataModule):

    def __init__(self, tokenizer: T5Tokenizer, args: Any):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args

        if self.args.task_name == "T2A":
            self.source_column = self.args.text_column
            self.target_column = self.args.aspects_column
        elif self.args.task_name == "A2P":
            self.source_column = self.args.aspects_column
            self.target_column = self.args.polarities_column
        else:
            raise ValueError("Invalid task name")
        
    def setup(self, stage=None):
        if stage == "fit":
            train_df = pd.read_csv(self.args.train_dataset_path)
            val_df = pd.read_csv(self.args.val_dataset_path)

            if self.args.verbose:
                print("Train dataset: ", train_df.head(2))

            self.train_dataset = T5ABSADataset(
                tokenizer=self.tokenizer,
                inputs=train_df[self.source_column].tolist(),
                targets=train_df[self.target_column].tolist(),
                aspects_annotations=train_df[self.args.annotations_column].tolist(),
                args=self.args
            )

            self.val_dataset = T5ABSADataset(
                tokenizer=self.tokenizer,
                inputs=val_df[self.source_column].tolist(),
                targets=val_df[self.target_column].tolist(),
                aspects_annotations=val_df[self.args.annotations_column].tolist(),
                args=self.args
            )
            
        if stage == "test":
            test_df = pd.read_csv(self.args.test_dataset_path)
            self.test_dataset = T5ABSADataset(
                tokenizer=self.tokenizer,
                inputs=test_df[self.source_column].tolist(),
                targets=test_df[self.target_column].tolist(),
                aspects_annotations=test_df[self.args.aspects_annotations_column].tolist(),
                args=self.args
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.tbatch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.vbatch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.vbatch_size)


class ABSAData(object):

    def __init__(
        self,
        aspects_categories: List[str],
        aspects_terms: Dict[str, List[str]],
        sentiment_polarities: Union[int, List[str]],
        oov_aspect_category: bool=False,
        oov_aspect_term: bool=False,
        oov_sentiment_polarity: bool=False
    ):
        self.aspects_categories = aspects_categories
        self.aspects_terms = aspects_terms
        self.sentiment_polarities = self._update_sentiment_polarities(sentiment_polarities)
        self.oov_aspect_category = oov_aspect_category
        self.oov_aspect_term = oov_aspect_term
        self.oov_sentiment_polarity = oov_sentiment_polarity

    def _update_sentiment_polarities(self, sentiment_polarities: Union[int, List[str]]) -> List[str]:
        if isinstance(sentiment_polarities, int):
            assert sentiment_polarities in {2, 3, 5}, "Invalid sentiment polarity value"
            if sentiment_polarities == 2:
                sentiment_polarities = ["positive", "negative"]
            elif sentiment_polarities == 3:
                sentiment_polarities = ["positive", "negative", "neutral"]
            elif sentiment_polarities == 5:
                sentiment_polarities = ["very positive", "positive", "neutral", "negative", "very negative"]
        return sentiment_polarities

