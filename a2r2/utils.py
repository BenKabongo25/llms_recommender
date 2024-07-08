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
from common.utils.vocabulary import Vocabulary, create_vocab_from_df, to_vocab_id