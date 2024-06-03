# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - May 2024

# Amazon Reviews 2023: loading and formatting data

import argparse
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def process_chunk(file_path, chunk_size, columns, output_path):
    with open(file_path, 'r') as fp:
        data = []
        for i, line in enumerate(tqdm(fp, desc=f"Processing {file_path}")):
            item = json.loads(line.strip())
            data.append(
                {k_column: item[v_column] for k_column, v_column in columns.items()}
            )

            if (i + 1) % chunk_size == 0:
                df = pd.DataFrame(data)
                df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False, escapechar='\\')
                data = []
        
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False, escapechar='\\')


def process_dataset(args):
    output_dir = os.path.join(args.output_base_dir, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    data_df_path = os.path.join(output_dir, "data.csv")
    if not os.path.exists(data_df_path):
        data_temp_path = os.path.join(output_dir, "data_temp.csv")
        columns = {
            "user_id": args.user_id_column, 
            "item_id": args.item_id_column, 
            "rating": args.rating_column, 
            "review": args.review_column, 
            "timestamp": args.timestamp_column
        }
        process_chunk(args.dataset_file, chunk_size=10000, columns=columns, output_path=data_temp_path)

        data_df = pd.read_csv(data_temp_path)
        data_df.to_csv(data_df_path, index=False)
        os.remove(data_temp_path)
    else:
        data_df = pd.read_csv(data_df_path)

    if args.verbose:
        print(args.dataset_name)
        print("Data:")
        print(data_df.sample(n=2))

    metadata_df_path = os.path.join(output_dir, "items.csv")
    if not os.path.exists(metadata_df_path):
        meta_temp_path = os.path.join(output_dir, "meta_temp.csv")
        columns = {
            "item_id": args.meta_item_id_column, 
            "description": args.meta_item_description_column
        }
        columns = [args.meta_item_id_column, args.meta_item_description_column]
        process_chunk(args.items_metadata_file, chunk_size=10000, columns=columns, output_path=meta_temp_path)

        metadata_df = pd.read_csv(meta_temp_path)
        metadata_df.to_csv(metadata_df_path, index=False)
        os.remove(meta_temp_path)
    else:
        metadata_df = pd.read_csv(metadata_df_path)

    if args.verbose:
        print("Metadata:")
        print(metadata_df.sample(n=2))
        print()

    user_reviews_count = data_df.groupby('user_id').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_reviews_count, bins=50, kde=True)
    plt.title('Number of reviews/ratings per user')
    plt.xlabel('Number of reviews/ratings')
    plt.ylabel('Number of users')
    plt.savefig(os.path.join(output_dir, "users_stats.png"))

    item_reviews_count = data_df.groupby('item_id').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(item_reviews_count, bins=50, kde=True)
    plt.title('Number of reviews/ratings per item')
    plt.xlabel('Number of reviews/ratings')
    plt.ylabel('Number of items')
    plt.savefig(os.path.join(output_dir, "items_stats.png"))

    plt.figure(figsize=(10, 6))
    sns.histplot(data_df['rating'], bins=5, kde=True)
    plt.title('Rating distribution')
    plt.xlabel('Rating')
    plt.ylabel('Number of reviews')
    plt.savefig(os.path.join(output_dir, "ratings_stats.png"))

    review_length = data_df['review'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(review_length, bins=50, kde=True)
    plt.title('Review length distribution')
    plt.xlabel('Review length')
    plt.ylabel('Number of reviews')
    plt.savefig(os.path.join(output_dir, "reviews_stats.png"))

    description_length = metadata_df['description'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(description_length, bins=50, kde=True)
    plt.title('Description length distribution')
    plt.xlabel('Description length')
    plt.ylabel('Number of descriptions')
    plt.savefig(os.path.join(output_dir, "descriptions_stats.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_file", type=str, default="")
    parser.add_argument("--items_metadata_file", type=str, default="")
    parser.add_argument("--output_base_dir", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    
    parser.add_argument("--user_id_column", type=str, default="user_id")
    parser.add_argument("--item_id_column", type=str, default="parent_asin")
    parser.add_argument("--rating_column", type=str, default="rating")
    parser.add_argument("--review_column", type=str, default="text")
    parser.add_argument("--timestamp_column", type=str, default="timestamp")

    parser.add_argument("--meta_item_id_column", type=str, default="parent_asin")
    parser.add_argument("--meta_item_description_column", type=str, default="title")
    
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    process_dataset(args)
