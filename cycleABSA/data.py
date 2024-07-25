# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: Data

import ast
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import T5Tokenizer
from typing import *

from annotations_text import AnnotationsTextFormerBase
from enums import TaskType
from prompts import PrompterBase
from utils import preprocess_text


class T5ABSADataset(Dataset):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        annotations_text_former: AnnotationsTextFormerBase,
        prompter: PrompterBase,
        data_df: pd.DataFrame,
        args: Any,
        task_type: Optional[TaskType]=None,
        split_name: str=""
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.annotations_text_former = annotations_text_former
        self.prompter = prompter

        self.texts = data_df[args.text_column].tolist()
        self.annotations = data_df[args.annotations_column].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        ).tolist()
        self.args = args
        self.task_type = task_type if task_type is not None else self.args.task_type
        self.split_name = split_name

        self.input_texts = []
        self.target_texts = []

        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        self._build()

    def __len__(self):
        return len(self.texts)

    def _build(self):
        desc = "Prepare data" if not self.split_name else self.split_name
        for idx in tqdm(range(len(self)), desc, colour="green"):
            text = preprocess_text(self.texts[idx], self.args)
            annotations = self.annotations[idx]
            annotations = [tuple([preprocess_text(t, self.args) for t in ann]) for ann in annotations]
            self.annotations[idx] = annotations

            annotations_text = self.annotations_text_former.multiple_annotations_to_text(annotations)

            if self.task_type is TaskType.T2A:
                input_text = text
                target_text = annotations_text
            else:
                input_text = annotations_text
                target_text = text

            input_text = self.prompter.get_prompt(
                task_type=self.task_type,
                text=input_text, 
                annotations=annotations, 
                polarities=self.annotations_text_former.absa_data.sentiment_polarities,
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
