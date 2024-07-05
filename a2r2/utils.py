# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Linear aspects model: Utils

import numpy as np
import pandas as pd
import os
import sys
import warnings
from typing import Any

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
warnings.filterwarnings(action="ignore")

from common.utils.evaluation import ratings_aspects_evaluation
from common.utils.functions import set_seed
from common.utils.vocabulary import Vocabulary


def process_data(data_df: pd.DataFrame, args: Any=None):
    users = data_df[args.user_id_column].unique()
    users_vocab = Vocabulary()
    users_vocab.add_elements(users)

    items = data_df[args.item_id_column].unique()
    items_vocab = Vocabulary()
    items_vocab.add_elements(items)

    def to_vocab_id(element, vocabulary: Vocabulary) -> int:
        return vocabulary.element2id(element)
    
    data_df[args.user_vocab_id_column] = data_df[args.user_id_column].apply(
        lambda u: to_vocab_id(u, users_vocab)
    )
    data_df[args.item_vocab_id_column] = data_df[args.item_id_column].apply(
        lambda i: to_vocab_id(i, items_vocab)
    )
    return data_df, users_vocab, items_vocab