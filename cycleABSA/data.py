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

from annotations_text import AnnotationsTextFormerBase
from enums import TaskType
from prompts import get_prompt
from utils import preprocess_text


class T5ABSADataset(Dataset):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        annotations_text_former: AnnotationsTextFormerBase, 
        data_df: pd.DataFrame,
        args: Any
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.annotations_text_former = annotations_text_former

        self.texts = data_df[args.text_column].tolist()
        self.annotations = data_df[args.annotations_column].apply(ast.literal_eval).tolist()
        self.args = args

        self.input_texts = []
        self.target_texts = []

        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        self._build()

    def __len__(self):
        return len(self.texts)

    def _build(self):
        for idx in tqdm(range(len(self)), "Prepare data", colour="green"):
            text = preprocess_text(self.texts[idx], self.args)
            annotations = self.annotations[idx]
            annotations = [tuple([preprocess_text(t, self.args) for t in ann]) for ann in annotations]
            self.annotations[idx] = annotations

            annotations_text = self.annotations_text_former.multiple_annotations_to_text(annotations)

            if self.args.task_type is TaskType.T2A:
                input_text = text
                target_text = annotations_text
            else:
                input_text = annotations_text
                target_text = text

            if self.args.prompt_flag:
                input_text = get_prompt(
                    text=input_text, 
                    annotations=annotations, 
                    polarities=self.annotations_text_former.absa_data.sentiment_polarities,
                    args=self.args
                )
            
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

            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.labels.append(labels)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        annotations = self.annotations[idx]

        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        labels = self.labels[idx]

        return {
            "input_texts": input_text,
            "target_texts": target_text,
            "annotations": annotations,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = [d[key] for d in batch]
        if isinstance(collated_batch[key][0], torch.Tensor):
            collated_batch[key] = torch.cat(collated_batch[key], 0)
    return collated_batch
        

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


class T5DataModule(pl.LightningDataModule):

    def __init__(self, tokenizer: T5Tokenizer, args: Any):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args

    def setup(self, stage=None):
        if stage == "fit":
            train_df = pd.read_csv(self.args.train_dataset_path)
            val_df = pd.read_csv(self.args.val_dataset_path)

            self.train_dataset = T5ABSADataset(
                tokenizer=self.tokenizer,
                data_df=train_df,
                args=self.args
            )

            self.val_dataset = T5ABSADataset(
                tokenizer=self.tokenizer,
                data_df=val_df,
                args=self.args
            )
            
        if stage == "test":
            test_df = pd.read_csv(self.args.test_dataset_path)
            self.test_dataset = T5ABSADataset(
                tokenizer=self.tokenizer,
                data_df=test_df,
                args=self.args
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.args.tbatch_size, shuffle=True, collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.args.vbatch_size, shuffle=False, collate_fn=collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.args.vbatch_size, shuffle=False, collate_fn=collate_fn
        )

