# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Basline approach
# Datasets and data split

import enum
import numpy as np
import os
import pandas as pd
from typing import *
from torch.utils.data import Dataset
from tqdm import tqdm

from prompters import SourcePrompter, TargetFormer
from sampler import Sampler, SamplingMethod, SimilarityFunction


class SplitMethod(enum.Enum):
    RANDOM = 0
    USER_BASED = 1
    ITEM_BASED = 2
    USER_ITEM_BASED = 3


class DataSplitter:

    def __init__(self, args):
        self.args = args
        self.split_method = SplitMethod(self.args.split_method)


    def split(self, data_df: pd.DataFrame) -> Dict:
        base_df = data_df.sample(
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

    
    def train_test_split(self, data_df: pd.DataFrame) -> Dict:
        if self.split_method is SplitMethod.USER_BASED:
            users = data_df[self.args.user_id_column].unique()
            n_users = len(users)
            n_train = int(self.args.train_size * n_users)
            train_users = np.random.choice(users, size=n_train, replace=False)
            train_df = data_df[data_df[self.args.user_id_column].isin(train_users)]
            test_df = data_df[~data_df[self.args.user_id_column].isin(train_users)]
        
        elif self.split_method is SplitMethod.ITEM_BASED:
            items = data_df[self.args.item_id_column].unique()
            n_items = len(items)
            n_train = int(self.args.train_size * n_items)
            train_items = np.random.choice(items, size=n_train, replace=False)
            train_df = data_df[data_df[self.args.item_id_column].isin(train_items)]
            test_df = data_df[~data_df[self.args.item_id_column].isin(train_items)]
        
        elif self.split_method is SplitMethod.USER_ITEM_BASED:
            users = data_df[self.args.user_id_column].unique()
            items = data_df[self.args.item_id_column].unique()
            n_users = len(users)
            n_items = len(items)
            n_train_users = int(self.args.train_size * n_users)
            n_train_items = int(self.args.train_size * n_items)
            train_users = np.random.choice(users, size=n_train_users, replace=False)
            train_items = np.random.choice(items, size=n_train_items, replace=False)
            train_df = data_df[
                (data_df[self.args.user_id_column].isin(train_users)) &
                (data_df[self.args.item_id_column].isin(train_items))
            ]
            test_df = data_df[
                (~data_df[self.args.user_id_column].isin(train_users)) |
                (~data_df[self.args.item_id_column].isin(train_items))
            ]

        else: # random
            train_df = data_df.sample(
                frac=self.args.train_size, 
                random_state=self.args.random_state
            )
            test_df = data_df.drop(train_df.index)

        return {
            "train": self.split(train_df),
            "test": self.split(test_df)
        }
    

class DatasetCreator:

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
        
        self.source_prompter = SourcePrompter(self.args)
        self.target_former = TargetFormer(self.args)
        
        self.sampled_data_df = None
        self.text_df = None


    def _add_sampled_data(self, samples_df: pd.DataFrame):
        if samples_df is None:
            return
        if self.sampled_data_df is None:
            self.sampled_data_df = samples_df
        else:
            self.sampled_data_df = pd.concat(
                [self.sampled_data_df, samples_df]
            ).drop_duplicates()

    
    def _add_text_data(self, text_df: pd.DataFrame):
        if text_df is None:
            return
        if self.text_df is None:
            self.text_df = text_df
        else:
            self.text_df = pd.concat(
                [self.text_df, text_df]
            ).drop_duplicates()


    def save_data(self, save_dir: str):
        if self.sampled_data_df is not None:
            self.sampled_data_df.to_csv(
                os.path.join(save_dir, "sampled_data.csv"),
                index=False
            )
        if self.text_df is not None:
            self.text_df.to_csv(
                os.path.join(save_dir, "text_data.csv"),
                index=False
            )


    def _format_example(self, row, user_based: bool) -> str:
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
        
        rating = sample[self.args.rating_column]
        review = TargetFormer.process_text(
            text=sample[self.args.review_column],
            max_length=self.args.max_review_length,
            args=self.args
        )

        return {
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "review": review,
            "user_examples": user_examples,
            "item_examples": item_examples,
            "user_description": user_description,
            "item_description": item_description,
        }
        

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

    
    def _zero_shot_add(self, index: int):
        sample, (user_examples, item_examples) = self._getitem(index)

        self._add_sampled_data(user_examples)
        self._add_sampled_data(item_examples)

        sample_ = self._format_sample(sample, user_examples, item_examples)
        source_text = self.source_prompter(sample_)
        target_text = self.target_former(sample_)

        text_dict = {}
        for k in ["user_id", "item_id", "review", "rating"]:
            text_dict[k] = [sample_[k]]
        text_dict["source"] = [source_text]
        text_dict["target"] = [target_text]
        text_data = pd.DataFrame(text_dict)
        self._add_text_data(text_data)

        
    def _few_shot_add(self, index: int):
        sample, (data, shots, ui_examples) = self._getitem(index)

        self._add_sampled_data(data)
        
        shots_samples = []
        for idx in data.index:
            vj_sample = data.loc[idx]
            v_examples, j_examples = shots[idx]

            self._add_sampled_data(v_examples)
            self._add_sampled_data(j_examples)

            shots_samples.append(self._format_sample(vj_sample, v_examples, j_examples))

        sample_ = {
            "shots": shots_samples,
            "sample": self._format_sample(sample, *ui_examples)
        }
        source_text = self.source_prompter(sample_)
        target_text = self.target_former(sample_)

        text_dict = {}
        for k in ["user_id", "item_id", "review", "rating"]:
            text_dict[k] = [sample_["sample"][k]]
        text_dict["source"] = [source_text]
        text_dict["target"] = [target_text]
        text_data = pd.DataFrame(text_dict)
        self._add_text_data(text_data)


    def create_dataset(self):
        for index in tqdm(
            range(len(self.base_df)), 
            desc="Dataset creation",
            colour="green"
        ):
            if self.sampler.is_zero_shot():
                self._zero_shot_add(index)
            else:
                self._few_shot_add(index)


    def get_sampled_df(self) -> pd.DataFrame:
        return self.sampled_data_df
    

    def get_text_df(self) -> pd.DataFrame:
        return self.text_df
    

class TextDataset(Dataset):

    def __init__(self, text_df: pd.DataFrame):
        super().__init__()
        self.text_df = text_df

    def __len__(self) -> int:
        return len(self.text_df)

    def __getitem__(self, index: int) -> Any:
        item = self.text_df.iloc[index]
        return {
            "source_text": item["source"],
            "target_text": item["target"],
            "review": item["review"],
            "rating": item["rating"]
        }
    
