# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Basline approach
# Similarities functions

import enum
import numpy as np
import pandas as pd
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
    r_b = np.array(common_data_df[args.rating_column + "_a"])

    if sim_fn is SimilarityFunction.COSINE:
        sim = (r_a * r_b).sum() / np.sqrt((r_a ** 2).sum() * (r_b ** 2).sum())
    elif sim_fn is SimilarityFunction.PEARSON:
        sim = (((r_a - r_a.mean()) * (r_b - r_b.mean())).sum() / 
                np.sqrt(((r_a - r_a.mean()) ** 2).sum() * ((r_b - r_b.mean()) ** 2).sum()))
    elif sim_fn is SimilarityFunction.MSD:
        sim = len(r_a) / ((r_a - r_b) ** 2).sum()
    return sim
