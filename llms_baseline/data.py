# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Basline approach
# Datasets and data split

import enum
import numpy as np
import pandas as pd
from typing import *
from torch.utils.data import Dataset

from prompters import SourcePrompter, TargetFormer
from sampler import Sampler, SamplingMethod
from similarities import SimilarityFunction


class SplitMethod(enum.Enum):
    RANDOM = 0
    USER_BASED = 1
    ITEM_BASED = 2
    USER_ITEM_BASED = 3


class DataSplitter:
    """
    Args:
        - split_method: int
            see data.SplitMethod
        - n_reviews
        - base_data_size: float
        - max_base_data_samples : int
        - train_size: float
        - test_size: float
        - val_size: float
        - user_id_column: str
        - item_id_column: str
    """

    def __init__(self, args):
        self.args = args
        self.split_method = SplitMethod(self.args.split_method)


    def filter_min_examples(self, data_df: pd.DataFrame, n_examples: int=1):
        users_count = (
            data_df
            .groupby(self.args.user_id_column)
            .size()
            .reset_index(name="n_examples")
        )
        users_filtered = (
            users_count[users_count["n_examples"] > n_examples][self.args.user_id_column]
        ).tolist()

        items_count = (
            data_df
            .groupby(self.args.item_id_column)
            .size()
            .reset_index(name="n_examples")
        )
        items_filtered = (
            items_count[items_count["n_examples"] > n_examples][self.args.item_id_column]
        ).tolist()

        filtered_df = data_df[data_df[self.args.user_id_column].isin(users_filtered)]
        filtered_df = filtered_df[filtered_df[self.args.item_id_column].isin(items_filtered)]
        return filtered_df


    def split(self, data_df: pd.DataFrame) -> Dict:
        filtered_df = self.filter_min_examples(data_df, self.args.n_reviews)

        base_df = filtered_df.sample(
            frac=self.args.base_data_size, 
            random_state=self.args.random_state
        )
        if len(base_df) > self.args.max_base_data_samples:
            base_df = base_df.head(n=self.args.max_base_data_samples)
        
        sampling_df = data_df.drop(base_df.index)

        return {
            "sampling": sampling_df,
            "base": base_df
        }

    
    def train_test_eval_split(self, data_df: pd.DataFrame) -> Dict:
        test_size = self.args.test_size / (self.args.test_size + self.args.val_size)
        eval_size = 1 - test_size
        
        if self.split_method is SplitMethod.USER_BASED:
            users = data_df[self.args.user_id_column].unique()
            n_users = len(users)
            n_train = self.args.train_size * n_users
            # TODO:
            train_df, test_df, eval_df = None, None, None
        
        elif self.split_method is SplitMethod.ITEM_BASED:
            pass
            # TODO:
            train_df, test_df, eval_df = None, None, None

        else: # random
            train_df = data_df.sample(
                frac=self.args.train_size, 
                random_state=self.args.random_state
            )
            test_eval_df = data_df.drop(train_df.index)
            if eval_size == 0:
                test_df = test_eval_df
                eval_df = None
            else:
                test_df = test_eval_df.sample(
                    frac=test_size,
                    random_state=self.args.random_state
                )
                eval_df = test_eval_df.drop(test_df.index)

        default_split = {"sampling": None, "base": None}
            
        return {
            "train": self.split(train_df) if train_df is not None else default_split,
            "test": self.split(test_df) if test_df is not None else default_split,
            "eval": self.split(eval_df) if eval_df is not None else default_split,
        }
    

class BaseDataset(Dataset):
    """
    Args:
        - n_reviews: int
        - n_samples: int
            0 for zero shot, >= 1 for few shot
        - sampling_method: int
            see sampler.SamplingMethod
        - user_id_column: str
        - item_id_column: str
        - rating_column: str
        - review_column: str
        - timestamp_flag: bool
        - timestamp_column: str
        - random_state: int
        - user_description_flag: bool
        - item_description_flag: bool
        - user_description_column: str
        - item_description_column: str
        - source_review_flag: bool
        - source_rating_flag: bool
        - target_review_flag: bool
        - target_rating_flag: bool
    """

    def __init__(
        self, 
        sampling_df: pd.DataFrame, 
        base_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame]=None,
        items_df: Optional[pd.DataFrame]=None,
        args=None
    ):
        super().__init__()
        self.sampler = Sampler(
            data_df=sampling_df, 
            n_reviews=args.n_reviews,
            n_samples=args.n_samples,
            sampling_method=SamplingMethod(args.sampling_method),
            similarity_function=SimilarityFunction(args.similarity_function),
            args=args
        )

        self.base_df = base_df
        self.users_df = users_df
        self.items_df = items_df
        self.args = args


    def __len__(self) -> int:
        return len(self.base_df)
    
    def _getitem(self, index: int) -> Tuple:
        sample = self.base_df.iloc[index]

        timestamp = None
        if self.args.timestamp_flag:
            timestamp = sample[self.args.timestamp_column]

        return sample, self.sampler.sample(
            user_id=sample[self.args.user_id_column],
            item_id=sample[self.args.item_id_column],
            timestamp=timestamp
        )
    

    def _format_example(self, row, user_based: bool):
        if user_based:
            description_flag = self.args.item_description_flag
            description_df = self.items_df
            id_column = self.args.item_id_column
            description_column = self.args.item_description_column
        else:
            description_flag = self.args.user_description_flag
            description_df = self.users_df
            id_column = self.args.user_id_column
            description_column = self.args.user_description_column

        example = ""
        if description_flag:
            assert description_df is not None
            descs = description_df.loc[
                description_df[id_column] == row[id_column], 
                description_column
            ].values

            description = ""
            if len(descs) > 0:
                description = TargetFormer.process_text(
                    text=str(descs[0]), 
                    max_length=self.args.max_description_length,
                    args=self.args
                )
            example += description
            
            if len(description) > 0:
                if self.args.source_review_flag or self.args.source_rating_flag:
                    example += " | "
            
        review = ""
        if self.args.source_review_flag:
            review = TargetFormer.process_text(
                text=str(row[self.args.review_column]), 
                max_length=self.args.max_review_length,
                args=self.args
            )

        rating = ""
        if self.args.source_rating_flag:
            rating = str(row[self.args.rating_column])

        if self.args.source_review_flag and self.args.source_rating_flag:
            example += f"{review} | {rating}"
        elif self.args.source_review_flag:
            example += review
        elif self.args.source_rating_flag:
            example += rating

        return example
    

    def _format_sample(
        self, 
        sample: Dict,
        user_examples: pd.DataFrame, 
        item_examples: pd.DataFrame
    ) -> Dict:
        
        sep_token = "\n"

        user_examples = sep_token.join(
            user_examples.apply(
                lambda row: self._format_example(row, user_based=True), 
                axis=1
            ).tolist()
        )

        if self.args.user_only_flag:
            item_examples = ""
        else:
            item_examples = sep_token.join(
                item_examples.apply(
                    lambda row: self._format_example(row, user_based=False), 
                    axis=1
                ).tolist()
            )

        user_id = sample[self.args.user_id_column]
        user_description = ""
        if self.args.user_description_flag:
            descs = self.users_df.loc[
                self.users_df[self.args.user_id_column] == user_id,
                self.args.user_description_column
            ].values
            if len(descs) > 0:
                user_description = TargetFormer.process_text(
                    text=str(descs[0]), 
                    max_length=self.args.max_description_length,
                    args=self.args
                )

        item_id = sample[self.args.item_id_column]
        item_description = ""
        if self.args.item_description_flag:
            descs = self.items_df.loc[
                self.items_df[self.args.item_id_column] == item_id,
                self.args.item_description_column
            ].values
            if len(descs) > 0:
                item_description = TargetFormer.process_text(
                    text=str(descs[0]), 
                    max_length=self.args.max_description_length,
                    args=self.args
                )

        return {
            "user_id": user_id,
            "item_id": item_id,
            "rating": sample[self.args.rating_column],
            "review": sample[self.args.review_column],
            "user_examples": user_examples,
            "item_examples": item_examples,
            "user_description": user_description,
            "item_description": item_description,
        }
        
    
    def zero_shot_getitem(self, index: int) -> Dict:
        sample, (user_examples, item_examples) = self._getitem(index)
        return self._format_sample(sample, user_examples, item_examples)


    def few_shot_getitem(self, index: int) -> Dict:
        sample, (data, shots, ui_examples) = self._getitem(index)
        
        shots_samples = []
        for idx in data.index:
            vj_sample = data.loc[idx]
            vj_examples = shots[idx]
            shots_samples.append(self._format_sample(vj_sample, *vj_examples))

        return {
            "shots": shots_samples,
            "sample": self._format_sample(sample, *ui_examples)
        }
    

    def __getitem__(self, index: int) -> Dict:
        if self.sampler.is_zero_shot():
            return self.zero_shot_getitem(index)
        return self.few_shot_getitem(index)
    

class TextDataset(BaseDataset):

    def __init__(
        self, 
        sampling_df: pd.DataFrame,
        base_df: pd.DataFrame, 
        users_df: Optional[pd.DataFrame] = None, 
        items_df: Optional[pd.DataFrame] = None, 
        args=None
    ):
        super().__init__(sampling_df, base_df, users_df, items_df, args)
        self.source_prompter = SourcePrompter(self.args)
        self.target_former = TargetFormer(self.args)


    def __getitem__(self, index: int) -> Any:
        sample = super().__getitem__(index)
        source_text = self.source_prompter.prompt(sample)
        target_text = self.target_former.format(sample)
        return {
            "source_text": source_text,
            "target_text": target_text,
            "review": sample["review"],
            "rating": sample["rating"]
        }
