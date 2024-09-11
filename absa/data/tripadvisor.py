# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# Cycle ABSA: TripAdvisor

import argparse
import json
import os
import pandas as pd
from tqdm import tqdm

aspects_categories = ["service", "cleanliness", "value", "sleep quality", "rooms", "location"]
aspects_categories_columns = ["Service", "Cleanliness", "Value", "Sleep Quality", "Rooms", "Location"]

def process_file(file_path):
    with open(file_path, 'r') as fp:
        data = json.load(fp)["Reviews"]

        formatted_data = []
        for sample in data:
            text = sample.get("Content", None)
            if text is None:
                continue

            aspects_annotations = sample.get("Ratings", [])
            formatted_aspects_annotations = []
            for i, aspect_category in enumerate(aspects_categories):
                aspect_category_column = aspects_categories_columns[i]
                sentiment_polarity_score = aspects_annotations.get(aspect_category_column, None)
                if sentiment_polarity_score is None:
                    continue
                sentiment_polarity_score = int(sentiment_polarity_score)
                pair = (aspect_category, sentiment_polarity_score)
                formatted_aspects_annotations.append(pair)

            if len(formatted_aspects_annotations) == 0:
                continue

            formatted_data.append(
                {
                    "text": text,
                    "annotations": formatted_aspects_annotations
                }
            )

        return formatted_data


def process_dataset(args):
    data = []
    for file in tqdm(os.listdir(args.raw_dataset_dir), colour="green"):
        file_path = os.path.join(args.raw_dataset_dir, file)
        data = process_file(file_path)
        data_df = pd.DataFrame(data)
        data_df.to_csv(
            args.data_path,
            mode='a', 
            header=not os.path.exists(args.data_path), 
            index=False, 
            escapechar='\\'
        )
        data = []

    data_df = pd.read_csv(args.data_path)
    if args.verbose:
        print(data_df.sample(n=2))
        print()

def main(args):
    if not os.path.exists(args.processed_dataset_dir):
        os.makedirs(args.processed_dataset_dir)
    args.data_path = os.path.join(args.processed_dataset_dir, "data.csv")
    process_dataset(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_dir", type=str, 
        default=os.path.join("Datasets", "raw", "TripAdvisor", "json"))
    parser.add_argument("--processed_dataset_dir", type=str,
        default=os.path.join("Datasets", "absa", "TripAdvisor"))
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    main(args)
