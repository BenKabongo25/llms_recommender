# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Basline approach
# Samplers for zero and few shot

import enum
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import *


class SimilarityFunction(enum.Enum):
    COSINE = 0
    PEARSON = 1
    MSD = 2


def compute_similarity(
    data_df: pd.DataFrame,
    a_id: Union[int, float, str],
    b_id: Union[int, float, str],
    user_based: bool=True,
    args=None
) -> float:
    sim_fn = SimilarityFunction(args.similarity_function)

    if user_based:
        first_column = args.user_id_column
        second_column = args.item_id_column
    else:
        first_column = args.item_id_column
        second_column = args.user_id_column

    df_a = data_df[data_df[first_column] == a_id]
    df_b = data_df[data_df[first_column] == b_id]

    common_data_df = pd.merge(
        df_a, df_b, on=second_column, how="inner", suffixes=["_a", "_b"]
    )
    
    r_a = np.array(common_data_df[args.rating_column + "_a"])
    r_b = np.array(common_data_df[args.rating_column + "_b"])

    if sim_fn is SimilarityFunction.COSINE:
        sim = (r_a * r_b).sum() / np.sqrt((r_a ** 2).sum() * (r_b ** 2).sum())
    elif sim_fn is SimilarityFunction.PEARSON:
        sim = (((r_a - r_a.mean()) * (r_b - r_b.mean())).sum() / 
                np.sqrt(((r_a - r_a.mean()) ** 2).sum() * ((r_b - r_b.mean()) ** 2).sum()))
    elif sim_fn is SimilarityFunction.MSD:
        sim = len(r_a) / ((r_a - r_b) ** 2).sum()
    else:
        raise ValueError("Unknown similarity function!")
    return sim


class SamplingMethod(enum.Enum):
    RANDOM = 0
    RATING_BASE = 1
    TIMESTAMP_BASE = 2
    SIMILARITY_BASE = 3


class Sampler:

    def __init__(
        self, 
        data_df: pd.DataFrame, 
        n_reviews: int,
        n_samples: int=0,
        sampling_method: SamplingMethod=SamplingMethod.RANDOM,
        similarity_function: SimilarityFunction=SimilarityFunction.COSINE,
        args=None
    ):
        self.data_df = data_df
        self.n_reviews = n_reviews
        self.n_samples = n_samples
        self.sampling_method = sampling_method
        self.similarity_function = similarity_function
        self.similarities = None
        self.args = args
        
        if self.sampling_method is SamplingMethod.SIMILARITY_BASE:
            self._compute_similarities()


    def _compute_similarities(self):
        def f(user_based):
            column = self.args.user_id_column if user_based else self.args.item_id_column
            elements = self.data_df[column].unique().tolist()
            n_elements = len(elements)

            similarities = {}
            tqdm_desc = ("Users" if user_based else "Items") + " similarities"
            for i in tqdm(range(0, n_elements), tqdm_desc, colour="orange"):
                a_id = elements[i]

                for j in range(i + 1, n_elements):
                    b_id = elements[j]
                    sim_ab = compute_similarity(
                        data_df=self.data_df,
                        a_id=a_id,
                        b_id=b_id,
                        user_based=user_based,
                        args=self.args
                    )

                    if a_id not in similarities:
                        similarities[a_id] = {}
                    if b_id not in similarities:
                        similarities[b_id] = {}

                    similarities[a_id][b_id] = sim_ab
                    similarities[b_id][a_id] = sim_ab
            
            return similarities
                
        users_similarities = f(user_based=True)
        items_similarities = f(user_based=False)

        self.similarities = {"users": users_similarities, "items": items_similarities}


    def _sample_from_df(
        self, 
        df: pd.DataFrame, 
        n_samples: int, 
        random_state: Union[None, int]
    ) -> pd.DataFrame:
        if n_samples > len(df):
            return df
        return df.sample(n=n_samples, random_state=random_state)
    

    def _sample(
        self,
        user_based: bool,
        id: Union[int, float, str],
        negative_ids: Optional[List]=[],
        timestamp: Optional[int]=None
    ) -> pd.DataFrame:
        if user_based:
            a_column = self.args.user_id_column
            b_column = self.args.item_id_column
        else:
            a_column = self.args.item_id_column
            b_column = self.args.user_id_column

        data = self.data_df[self.data_df[a_column] == id]

        if negative_ids is not None or len(negative_ids) > 0:
            data = data[~data[b_column].isin(negative_ids)]

        if self.sampling_method is SamplingMethod.RATING_BASE:
            min_rating = np.min(data[self.args.rating_column])
            max_rating = np.max(data[self.args.rating_column])
            mean_rating = (min_rating + max_rating) / 2

            negatives_data = data[data[self.args.rating_column] < mean_rating]
            positives_data = data[data[self.args.rating_column] >= mean_rating]

            negatives_samples = self._sample_from_df(
                negatives_data,
                n_samples=self.n_reviews // 2, 
                random_state=self.args.random_state
            )
            positives_samples = self._sample_from_df(
                positives_data,
                n_samples=self.n_reviews // 2, 
                random_state=self.args.random_state
            )

            samples = pd.concat([negatives_samples, positives_samples])

        elif timestamp is not None and self.sampling_method is SamplingMethod.TIMESTAMP_BASE:
            data = data[data[self.args.timestamp_column] < timestamp]
            samples = self._sample_from_df(
                data,
                n_samples=self.n_reviews, 
                random_state=self.args.random_state
            )

        elif negative_ids is not None and self.sampling_method is SamplingMethod.SIMILARITY_BASE:
            key = "items" if user_based else "users"
            samples_ids = []
            
            for n_id in negative_ids:
                similarities = self.similarities[key][n_id]
                candidates_ids, candidates_similarities = similarities.items()
                candidates_ids = np.array(candidates_ids)
                candidates_similarities = np.array(candidates_similarities)
                argsort = np.argsort(candidates_similarities)[::-1]
                sorted_candidates_ids = candidates_ids[argsort]
                samples_ids.extend(sorted_candidates_ids)

            samples_ids = samples_ids[:self.n_reviews]
            samples = data[data[b_column].isin(samples_ids)]
            
        else: # random sampling
            samples = self._sample_from_df(
                data,
                n_samples=self.n_reviews, 
                random_state=self.args.random_state
            )
        return samples
    

    def user_sample(
        self, 
        user_id: Union[int, float, str],
        negative_items_ids: Optional[List]=[],
        timestamp: Optional[int]=None
    ) -> pd.DataFrame:
        return self._sample(
            user_based=True, 
            id=user_id, 
            negative_ids=negative_items_ids, 
            timestamp=timestamp
        )
         
    
    def item_sample(
        self, 
        item_id: Union[int, float, str],
        negative_users_ids: Optional[List]=[],
        timestamp: Optional[int]=None
    ) -> pd.DataFrame:
        return self._sample(
            user_based=False, 
            id=item_id, 
            negative_ids=negative_users_ids, 
            timestamp=timestamp
        )
         

    def zero_shot_sample(
        self,
        user_id: Union[int, float, str],
        item_id: Union[int, float, str],
        timestamp: Optional[int]=None
    ) -> Tuple[pd.DataFrame]:
        user_examples = self.user_sample(
            user_id=user_id,
            negative_items_ids=[item_id],
            timestamp=timestamp
        )
        
        item_examples = None
        if not self.args.user_only_flag:
            item_examples = self.item_sample(
                item_id=item_id,
                negative_users_ids=[user_id],
                timestamp=timestamp
            )

        return user_examples, item_examples


    def few_shot_sample(
        self,
        user_id: Union[int, float, str],
        item_id: Union[int, float, str],
        timestamp: Optional[int]=None
    ) -> Tuple[pd.DataFrame]:
        data = self._sample_from_df(
            self.data_df[(self.data_df[self.args.user_id_column] != user_id) & 
                         (self.data_df[self.args.item_id_column] != item_id)],
            n_samples=self.n_samples,
            random_state=self.args.random_state
        )

        samples = dict()
        for idx, row in data.iterrows():
            v_id = row[self.args.user_id_column]
            j_id = row[self.args.item_id_column]
            t = None
            if self.args.timestamp_flag:
                t = row[self.args.timestamp_column]
            samples[idx] = self.zero_shot_sample(user_id=v_id, item_id=j_id, timestamp=t)

        ui_examples = self.zero_shot_sample(
            user_id=user_id, 
            item_id=item_id, 
            timestamp=timestamp
        )

        return data, samples, ui_examples
    

    def is_zero_shot(self) -> bool:
        return self.n_samples == 0
    

    def sample(
        self,
        user_id: Union[int, float, str],
        item_id: Union[int, float, str],
        timestamp: Optional[int]=None
    ) -> Tuple[pd.DataFrame]:
        if self.is_zero_shot():
            return self.zero_shot_sample(user_id=user_id, item_id=item_id, timestamp=timestamp)
        return self.few_shot_sample(user_id=user_id, item_id=item_id, timestamp=timestamp)
        