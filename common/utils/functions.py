# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Common
# Utils functions

import numpy as np
import random
import time
import torch


def set_seed(args):
    args.time_id = int(time.time())
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
