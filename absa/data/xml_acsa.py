# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# Cycle ABSA: XML ACSA Datasets

import argparse
import os
import pandas as pd
import xml.etree.ElementTree as ET

def process_xml_to_dataframe(file_path, args):
    tree = ET.parse(file_path)
    root = tree.getroot()

    texts = []
    annotations = []

    for sentence in root.findall('sentence'):
        text = sentence.find('text').text
        aspect_categories = sentence.find('aspectCategories')
        
        annotation_list = []

        for aspectCategory in aspect_categories.findall('aspectCategory'):
            category = aspectCategory.get('category')
            polarity = aspectCategory.get('polarity')
            annotation_list.append((category, polarity))
        
        texts.append(text)
        annotations.append(annotation_list)

    data_df = pd.DataFrame({'text': texts, 'annotations': annotations})
    data_df.to_csv(
        args.data_path,
        mode='a',
        header=not os.path.exists(args.data_path), 
        index=False, 
        escapechar='\\'
    )


def main(args):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    args.data_path = os.path.join(args.data_dir, "data.csv")
    if os.path.exists(args.data_path):
        os.remove(args.data_path)

    for file in os.listdir(args.raw_dataset_dir):
        if file.endswith(".xml"):
            file_path = os.path.join(args.raw_dataset_dir, file)
            process_xml_to_dataframe(file_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_dir", type=str,
        default=os.path.join("Datasets", "raw", "ABSA", "MAMS", "ACSA"))
    parser.add_argument("--data_dir", type=str,
        default=os.path.join("Datasets", "absa", "mams_acsa"))
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    main(args)
