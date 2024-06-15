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
        
        if len(interactions) < args.min_reviews:
            return
        
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
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    items = []
    interactions = []
    for item_file in tqdm(os.listdir(args.dataset_dir)):
        item_file_path = os.path.join(args.dataset_dir, item_file)
        item_data = process_item(item_file_path, args)
        if item_data is None:
            continue
        item, item_interactions = item_data
        items.append(item)
        interactions.extend(item_interactions)

    items_df = pd.DataFrame(items)
    items_df_path = os.path.join(output_dir, "items.csv")
    items_df.to_csv(items_df_path, index=False)

    interactions_df = pd.DataFrame(interactions)
    interactions_df_path = os.path.join(output_dir, "data.csv")
    interactions_df.to_csv(interactions_df_path, index=False)

    if args.verbose:
        print("Items:")
        print(items_df.sample(n=2))
        print()
        print("Reviews:")
        print(interactions_df.sample(n=2))
        print()

    user_interactions_count = interactions_df.groupby('user_id').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_interactions_count, bins=50, kde=True)
    plt.title('Number of interactions/ratings per user')
    plt.xlabel('Number of interactions/ratings')
    plt.ylabel('Number of users')
    plt.savefig(os.path.join(output_dir, "users_stats.png"))

    item_interactions_count = interactions_df.groupby('item_id').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(item_interactions_count, bins=50, kde=True)
    plt.title('Number of interactions/ratings per item')
    plt.xlabel('Number of interactions/ratings')
    plt.ylabel('Number of items')
    plt.savefig(os.path.join(output_dir, "items_stats.png"))



