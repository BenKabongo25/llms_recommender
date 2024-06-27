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

from common.utils.evaluation import ratings_evaluation
from common.utils.functions import set_seed
from common.utils.vocabulary import Vocabulary


def rescale(x, a, b, c, d):
    return c + (d - c) * ((x - a) / (b - a))

aspect_scale = lambda x, args: rescale(
    x,
    args.min_rating, args.max_rating,
    args.aspect_min_rating, args.aspect_max_rating
)

global_scale = lambda x, args: rescale(
    x,
    args.aspect_min_rating, args.aspect_max_rating,
    args.min_rating, args.max_rating
)


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


def make_toy_dataset():
    n_users = 100
    n_items = 200
    n_aspects = 5
    
    users = [f"u{i}" for i in range(1, n_users + 1)]
    items = [f"i{i}" for i in range(1, n_items + 1)]
    aspects = [f"a{i}" for i in range(1, n_aspects + 1)]

    users_params = np.random.random(size=(n_users, n_aspects))
    users_params = users_params / users_params.sum(axis=1)[:, np.newaxis]

    aspects_ratings = np.random.randint(1, 6, (n_users, n_items, n_aspects))

    data = []
    for ui, u in enumerate(users):
        for ii, i in enumerate(items):
            row = {}
            row["user_id"] = u
            row["item_id"] = i
            r_ui = 0
            for ai, a in enumerate(aspects):
                row[a] = aspects_ratings[ui, ii, ai]
                r_ui += users_params[ui, ai] * aspects_ratings[ui, ii, ai]
            row["rating"] = r_ui
            data.append(row)
    data_df = pd.DataFrame(data)

    return data_df, aspects