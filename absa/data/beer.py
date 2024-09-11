# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# Cycle ABSA: Beer

import argparse
import json
import os
import pandas as pd
from tqdm import tqdm


aspects_categories = ["appearance", "aroma", "palate", "taste"]
aspects_categories_columns = [
    "review/appearance", "review/aroma", "review/palate", "review/taste"
]

def rescale(x, a, b, c, d):
    return c + (d - c) * ((x - a) / (b - a))


def process_dataset(args):
    data = []
    with open(args.dataset_file, 'r', encoding="utf-8") as fp:
        for line in tqdm(fp, args.dataset_name, colour="green", total=args.n_lines):
            try:
                sample = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if not bool(sample):
                continue
            
            text = sample.get("review/text", None)
            if text is None:
                continue

            formatted_aspects_annotations = []
            for i, aspect_category in enumerate(aspects_categories):
                aspect_category_column = aspects_categories_columns[i]
                sentiment_polarity_score = sample.get(aspect_category_column, None)
                if sentiment_polarity_score is None:
                    continue
                num, denom = sentiment_polarity_score.split("/")
                sentiment_polarity_score = int(rescale(float(num), 0, float(denom), 1, 5))
                pair = (aspect_category, sentiment_polarity_score)
                formatted_aspects_annotations.append(pair)

            if len(formatted_aspects_annotations) == 0:
                continue

            data.append(
                {
                    "text": text,
                    "annotations": formatted_aspects_annotations
                }
            )

    data_df = pd.DataFrame(data)
    data_df.to_csv(args.data_path, index=False, escapechar='\\')
    if args.verbose:
        print(data_df.sample(n=2))


def main(args):
    args.data_dir = os.path.join(args.processed_dataset_dir, args.dataset_name)
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    args.data_path = os.path.join(args.data_dir, "data.csv")
    process_dataset(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str,
        default="Datasets\\raw\\Beer\\ratebeer_.json")
    parser.add_argument("--processed_dataset_dir", type=str,
        default=os.path.join("Datasets", "absa"))
    parser.add_argument("--dataset_name", type=str, default="RateBeer")
    parser.add_argument("--n_lines", type=int, default=2_800_000)    
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    main(args)
    