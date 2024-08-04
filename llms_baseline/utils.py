# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Basline approach
# Utils

import os
import sys
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from common.utils.evaluation import evaluate_fn, ratings_evaluation, reviews_evaluation
from common.utils.preprocess_text import preprocess_text
from common.utils.functions import set_seed


def empty_cache():
    with torch.no_grad(): 
        torch.cuda.empty_cache()


def collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = [d[key] for d in batch]
        if isinstance(collated_batch[key][0], torch.Tensor):
            collated_batch[key] = torch.cat(collated_batch[key], 0)
    return collated_batch
