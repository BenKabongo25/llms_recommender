# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# P5: https://arxiv.org/abs/2203.13366

import enum
import pandas as pd
import os
import sys
from tqdm import tqdm
from typing import *
from torch.utils.data import Dataset


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from common.utils.preprocess_text import preprocess_text


class PromptType(enum.Enum):
    RATING_1 = 0
    RATING_2 = 1
    REVIEW = 2


class SourcePrompter:

    def __init__(self, args):
        self.args = args
        self.prompt_type = PromptType(args.prompt_type)

    def prompt(self, sample: dict) -> str:
        user_id = sample[self.args.user_id_column]
        item_id = sample[self.args.item_id_column]
        item_description = sample[self.args.item_description_column]

        if self.prompt_type is PromptType.RATING_1:
            text = (
                f"Which star rating will user_[{user_id}] give item_[{item_id}]? "
                f"(1 being lowest and 5 being highest)"
            )
        elif self.prompt_type is PromptType.RATING_2:
            text = (
                f"How will user_[{user_id}] rate this product: [{item_description}]? "
                f"(1 being lowest and 5 being highest)"
            )
        elif self.prompt_type is PromptType.REVIEW:
            text = (
                f"Generate an explanation for user_[{user_id}] about this product: [{item_description}]"
            )
        else:
            text = ""
        
        return text


class TargetFormer:

    def __init__(self, args):
        self.args = args
        self.prompt_type = PromptType(args.prompt_type)

    def target(self, sample: dict) -> str:
        if self.prompt_type is PromptType.RATING_1:
            target = sample[self.args.rating_column]
        elif self.prompt_type is PromptType.RATING_2:
            target = sample[self.args.rating_column]
        elif self.prompt_type is PromptType.REVIEW:
            target = preprocess_text(
                text=sample[self.args.review_column], 
                args=self.args,
                max_length=self.args.max_review_length
            )
        else:
            target = ""
        
        return str(target)


class P5DataCreator:

    def __init__(self, args):
        self.args = args
        self.source_prompter = SourcePrompter(args)
        self.target_former = TargetFormer(args)

    def create_dataset(
        self, 
        data_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame]=None,
        items_df: Optional[pd.DataFrame]=None
    ) -> pd.DataFrame: 
        data = []
        for i, sample in tqdm(
            data_df.iterrows(),
            desc="Dataset creation",
            colour="green",
        ):
            sample = sample.to_dict()
            item_description = ""
            if items_df is not None:
                descs = items_df[
                    items_df[self.args.item_id_column] == sample[self.args.item_id_column]
                ][self.args.item_description_column].values
                if len(descs) > 0:
                    item_description = descs[0]
                    item_description = preprocess_text(
                        text=item_description, 
                        args=self.args,
                        max_length=self.args.max_description_length
                    )
            sample[self.args.item_description_column] = item_description

            source = self.source_prompter.prompt(sample)
            target = self.target_former.target(sample)
            data.append({
                "user_id": sample[self.args.user_id_column],
                "item_id": sample[self.args.item_id_column],
                "source": source,
                "target": target
            })
        p5_data_df = pd.DataFrame(data)
        return p5_data_df
    

class P5Dataset(Dataset):

    def __init__(self, data_df, args):
        self.data_df = data_df
        self.args = args

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        return {
            "user_id": sample["user_id"],
            "item_id": sample["item_id"],
            "source": sample["source"],
            "target": sample["target"]
        }
        
