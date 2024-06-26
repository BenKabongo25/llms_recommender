# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# TripAdvisor

import argparse
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import *


def process_item(item_file_path, args):
    with open(item_file_path, 'r') as fp:
        data = json.load(fp)

        item_infos = data['HotelInfo']
        interactions = data["Reviews"]

        item_id = item_infos.get("HotelID", None)
        item_name = item_infos.get("Name", None)
        item_price = item_infos.get("Price", None)

        if item_id is None or item_name is None:
            return
        
        item_id = item_id.replace(" ", "_")
        item_description = item_name
        if item_price is not None:
            item_description += f"; price {item_price}"
        
        formatted_interactions = []
        for review in interactions:
            user_id = review.get("Author", None)
            review_title = review.get("Title", None)
            review_text = review.get("Content", None)
            ratings = review.get("Ratings", {})
            rating = ratings.get("Overall", None)

            if user_id is None or review_text is None or rating is None:
                continue
            
            service = ratings.get("Service", None)
            cleanliness = ratings.get("Cleanliness", None)
            value = ratings.get("Value", None)
            sleep_quality = ratings.get("Sleep Quality", None)
            rooms = ratings.get("Rooms", None)
            location = ratings.get("Location", None)

            timestamp = review.get("Date", None)
            if timestamp is not None:
                timestamp = pd.to_datetime(timestamp).timestamp()

            formatted_interactions.append({
                "user_id": user_id,
                "item_id": item_id,
                "rating": rating,
                "review": review_text,
                "review_title": review_title,
                "timestamp": timestamp,
                "service": service,
                "cleanliness": cleanliness,
                "value": value,
                "sleep_quality": sleep_quality,
                "rooms": rooms,
                "location": location
            })

        item_data = {
            "item_id": item_id,
            "item_name": item_name,
            "item_description": item_description,
            "item_price": item_price
        }

        return item_data, formatted_interactions


def process_dataset(args):
    items = []
    interactions = []
    for item_file in tqdm(os.listdir(args.raw_dataset_dir), colour="green"):
        item_file_path = os.path.join(args.raw_dataset_dir, item_file)
        item_data = process_item(item_file_path, args)
        if item_data is None:
            continue

        item, item_interactions = item_data
        items.append(item)
        interactions.extend(item_interactions)

        items_df = pd.DataFrame(items)
        items_df.to_csv(
            args.items_path,
            mode='a', 
            header=not os.path.exists(args.items_path), 
            index=False, 
            escapechar='\\'
        )
        items = []

        data_df = pd.DataFrame(item_interactions)
        data_df.to_csv(
            args.data_path,
            mode='a', 
            header=not os.path.exists(args.data_path), 
            index=False, 
            escapechar='\\'
        )
        interactions = []

    items_df = pd.read_csv(args.items_path)
    data_df = pd.read_csv(args.data_path)

    if args.verbose:
        print("Items:")
        print(items_df.sample(n=2))
        print()
        print("Reviews:")
        print(data_df.sample(n=2))
        print()

    user_interactions_count = data_df.groupby('user_id').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_interactions_count, bins=50, kde=True)
    plt.title('Number of interactions/ratings per user')
    plt.xlabel('Number of interactions/ratings')
    plt.ylabel('Number of users')
    plt.savefig(os.path.join(args.processed_dataset_dir, "users_stats.png"))

    item_interactions_count = data_df.groupby('item_id').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(item_interactions_count, bins=50, kde=True)
    plt.title('Number of interactions/ratings per item')
    plt.xlabel('Number of interactions/ratings')
    plt.ylabel('Number of items')
    plt.savefig(os.path.join(args.processed_dataset_dir, "items_stats.png"))

    plt.figure(figsize=(10, 6))
    sns.histplot(data_df['rating'], bins=5, kde=True)
    plt.title('Rating distribution')
    plt.xlabel('Rating')
    plt.ylabel('Number of reviews')
    plt.savefig(os.path.join(args.processed_dataset_dir, "ratings_stats.png"))

    review_length = data_df['review'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(review_length, bins=50, kde=True)
    plt.title('Review length distribution')
    plt.xlabel('Review length')
    plt.ylabel('Number of reviews')
    plt.savefig(os.path.join(args.processed_dataset_dir, "reviews_stats.png"))

    description_length = items_df['item_description'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(description_length, bins=50, kde=True)
    plt.title('Description length distribution')
    plt.xlabel('Description length')
    plt.ylabel('Number of descriptions')
    plt.savefig(os.path.join(args.processed_dataset_dir, "descriptions_stats.png"))


def main(args):
    if not os.path.exists(args.processed_dataset_dir):
        os.makedirs(args.processed_dataset_dir)

    args.items_path = os.path.join(args.processed_dataset_dir, "items.csv")
    args.data_path = os.path.join(args.processed_dataset_dir, "data.csv")

    process_dataset(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--raw_dataset_dir", type=str, default="Datasets\\raw\\TripAdvisor\\json")
    parser.add_argument("--processed_dataset_dir", type=str, default="Datasets\\processed\\TripAdvisor")
    
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    main(args)
