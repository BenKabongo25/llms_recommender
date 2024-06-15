# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Data sampling and splitting


import argparse
import numpy as np
import os
import pandas as pd
import random
from typing import *


def sample_split(
	data_df: pd.DataFrame,
	users_df: Optional[pd.DataFrame]=None,
	items_df: Optional[pd.DataFrame]=None,
	args: Any=None
):
	all_users = data_df[args.user_id_column].value_counts()
	if users_df is not None:
		all_users = all_users[all_users.index.isin(users_df[args.user_id_column])]
	selected_users = all_users[all_users.values > args.min_interactions]
	if len(selected_users) > args.max_n_users:
		selected_users = selected_users[:args.max_n_users]
	data_df = data_df[
		data_df[args.user_id_column].isin(selected_users.index)
	]

	all_items = data_df[args.item_id_column].value_counts() 
	if items_df is not None:
		all_items = all_items[all_items.index.isin(items_df[args.item_id_column])]
	selected_items = all_items[all_items.values > args.min_interactions]
	if len(selected_items) > args.max_n_items:
		selected_items = selected_items[:args.max_n_items]
	data_df = data_df[
		data_df[args.item_id_column].isin(selected_items.index)
	]

	all_users = data_df[args.user_id_column].unique()
	if users_df is not None:
		users_df = users_df[users_df[args.user_id_column].isin(all_users)]
	else:
		users_df = pd.DataFrame({"user_id": all_users})
	n_users = len(all_users)
	n_seen_users = int(args.seen_size * n_users)
	seen_users = np.random.choice(all_users, size=n_seen_users, replace=False)
	unseen_users = np.setdiff1d(all_users, seen_users)
	n_unseen_users = len(unseen_users)
	unseen_users_group_1 = unseen_users[:n_unseen_users // 2]
	unseen_users_group_2 = unseen_users[n_unseen_users // 2:]
	seen_users_group_1 = np.setdiff1d(all_users, unseen_users_group_2)

	all_items = data_df[args.item_id_column].unique()
	if items_df is not None:
		items_df = items_df[items_df[args.item_id_column].isin(all_items)]
	else:
		items_df = pd.DataFrame({"item_id": all_items})
	n_items = len(all_items)
	n_seen_items = int(args.seen_size * n_items)
	seen_items = np.random.choice(all_items, size=n_seen_items, replace=False)
	unseen_items = np.setdiff1d(all_items, seen_items)
	n_unseen_items = len(unseen_items)
	unseen_items_group_1 = unseen_items[:n_unseen_items // 2]
	unseen_items_group_2 = unseen_items[n_unseen_items // 2:]
	seen_items_group_1 = np.setdiff1d(all_items, unseen_items_group_2)

	seen_df = data_df[
        (data_df[args.user_id_column].isin(seen_users_group_1)) &
        (data_df[args.item_id_column].isin(seen_items_group_1))
    ]
	train_seen_df = seen_df.sample(
		frac=args.train_seen_size, 
		random_state=args.random_state
	)
	test_seen_df = seen_df.drop(train_seen_df.index)

	unseen_pairs_df = data_df[
        (~data_df[args.user_id_column].isin(seen_users_group_1)) |
        (~data_df[args.item_id_column].isin(seen_items_group_1))
    ]
	unseen_users_df = data_df[data_df[args.user_id_column].isin(unseen_users)]
	unseen_items_df = data_df[data_df[args.item_id_column].isin(unseen_items)]

	if args.verbose:
		if users_df is not None:
			print("Sample of users_df:")
			print(users_df.sample(5))
		if items_df is not None:
			print("Sample of items_df:")
			print(items_df.sample(5))
		print("Sample of seen_df:")
		print(seen_df.sample(5))
		print("Sample of train_seen_df:")
		print(train_seen_df.sample(5))
		print("Sample of test_seen_df:")
		print(test_seen_df.sample(5))
		print("Sample of unseen_pairs_df:")
		print(unseen_pairs_df.sample(5))
		print("Sample of unseen_users_df:")
		print(unseen_users_df.sample(5))
		print("Sample of unseen_items_df:")
		print(unseen_items_df.sample(5))

	save_dir = os.path.join(args.dataset_dir, "samples")
	metadata_dir = os.path.join(save_dir, "metadata")
	splits_dir = os.path.join(save_dir, "splits") 
	seen_dir = os.path.join(splits_dir, "seen")
	unseen_dir = os.path.join(splits_dir, "unseen")
	os.makedirs(metadata_dir, exist_ok=True)
	os.makedirs(seen_dir, exist_ok=True)
	os.makedirs(unseen_dir, exist_ok=True)

	if users_df is not None:
		users_df.to_csv(os.path.join(metadata_dir, "users.csv"), index=False)
	if items_df is not None:
		items_df.to_csv(os.path.join(metadata_dir, "items.csv"), index=False)

	train_seen_df.to_csv(os.path.join(seen_dir, "train.csv"), index=False)
	test_seen_df.to_csv(os.path.join(seen_dir, "test.csv"), index=False)
	unseen_users_df.to_csv(os.path.join(unseen_dir, "users.csv"), index=False)
	unseen_items_df.to_csv(os.path.join(unseen_dir, "items.csv"), index=False)
	unseen_pairs_df.to_csv(os.path.join(unseen_dir, "pairs.csv"), index=False)


def main(args):
	random.seed(args.random_state)
	np.random.seed(args.random_state)

	if args.dataset_dir == "":
		args.dataset_dir = os.path.join(args.base_dir, args.dataset_name)

	data_path = os.path.join(args.dataset_dir, "data.csv")
	data_df = pd.read_csv(data_path).dropna().drop_duplicates()

	users_path = os.path.join(args.dataset_dir, "users.csv")
	users_df = None
	if os.path.exists(users_path):
		users_df = pd.read_csv(users_path).dropna().drop_duplicates()
	
	items_path = os.path.join(args.dataset_dir, "items.csv")
	items_df = None
	if os.path.exists(items_path):
		items_df = pd.read_csv(items_path).dropna().drop_duplicates()

	sample_split(data_df, users_df, items_df, args)
	 

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--base_dir", type=str, default="Datasets\\AmazonReviews2023_processed")
	parser.add_argument("--dataset_name", type=str, default="CDs_and_Vinyl")
	parser.add_argument("--dataset_dir", type=str, default="")
	
	parser.add_argument("--user_id_column", type=str, default="user_id")
	parser.add_argument("--item_id_column", type=str, default="item_id")
	parser.add_argument("--rating_column", type=str, default="rating")
	parser.add_argument("--review_column", type=str, default="review")
	parser.add_argument("--timestamp_column", type=str, default="timestamp")

	parser.add_argument("--min_interactions", type=int, default=5)
	parser.add_argument("--max_n_users", type=int, default=1_000_000)
	parser.add_argument("--max_n_items", type=int, default=1_000_000)
	parser.add_argument("--seen_size", type=float, default=0.9)
	parser.add_argument("--train_seen_size", type=float, default=0.8)
	parser.add_argument("--random_state", type=int, default=42)
	
	parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
	parser.set_defaults(verbose=True)
	args = parser.parse_args()

	main(args)
