# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Basline approach
# Utils

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from common.utils.evaluation import evaluate_fn, ratings_evaluation, reviews_evaluation
from common.utils.preprocess_text import preprocess_text
from common.utils.functions import set_seed
