# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# Cycle ABSA: Text-Sep-Annotations Datasets

import argparse
import os
import pandas as pd


def process_file(file_path, args):
    with open(file_path, 'r') as fp:
        lines = fp.readlines()

    data = []
    for line in lines:
        text, annotations = line.split(args.separator)
        text = text.strip()
        annotations = list(map( tuple, eval(annotations.strip())))
        data.append({"text": text, "annotations": annotations})

    data_df = pd.DataFrame(data)
    data_df.to_csv(
        args.data_path,
        mode='a',
        header=not os.path.exists(args.data_path), 
        index=False, 
        escapechar='\\'
    )
    if args.verbose:
        print(data_df.sample(n=2))


def main(args):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    args.data_path = os.path.join(args.data_dir, "data.csv")
    if os.path.exists(args.data_path):
        os.remove(args.data_path)

    for file in os.listdir(args.raw_dataset_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(args.raw_dataset_dir, file)
            process_file(file_path, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_dir", type=str,
        default=os.path.join("Datasets", "raw", "ABSA", "QUAD", "rest16"))
    parser.add_argument("--data_dir", type=str,
        default=os.path.join("Datasets", "absa", "Rest16"))
    parser.add_argument("--separator", type=str, default="####")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    main(args)
