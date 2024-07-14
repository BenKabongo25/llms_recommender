# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# CycleABSA: Utils

# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Basline approach
# Utils


import nltk
import numpy as np
import random
import re
import time
import torch

from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from typing import Any, List, Tuple


EXTRACTION_STYLE_ELEMENTS_SEPARATOR = ";"
EXTRACTION_STYLE_ELEMENT_BEGIN = "("
EXTRACTION_STYLE_ELEMENT_END = ")"
EXTRACTION_STYLE_ELEMENT_SEPARATOR = ","

ANNOTATION_STYLE_ELEMENT_BEGIN = "["
ANNOTATION_STYLE_ELEMENT_END = "]"
ANNOTATION_STYLE_ELEMENT_SEPARATOR = "|"


def get_elements_from_extraction_style(text: str) -> List[Tuple[str]]:
    elements = []
    for element_text in text.split(";"):
        element_text = element_text.strip()
        if not element_text.startswith("(") and element_text.endswith(")"):
            continue
        element_text = element_text[1:-1]
        element = tuple(element_text.split(","))
        elements.append(element)
    return elements


def get_elements_from_annotation_style(text: str) -> List[Tuple[str]]:
    elements = []
    # TODO: Implement the function
    return elements


def get_annotations(text: str, args: Any) -> List[Tuple[str]]:
    if args.annotation_style == "extraction":
        return get_elements_from_extraction_style(text)
    elif args.annotation_style == "annotation":
        return get_elements_from_annotation_style(text)
    else:
        raise ValueError(f"Unknown annotation style: {args.annotation_style}")


def set_seed(args):
    args.time_id = int(time.time())
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)


def delete_punctuation(text: str) -> str:
    punctuation = r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\n\t]"
    text = re.sub(punctuation, " ", text)
    text = re.sub('( )+', ' ', text)
    return text


def delete_stopwords(text: str) -> str:
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return ' '.join([w for w in text.split() if w not in stop_words])


def delete_non_ascii(text: str) -> str:
    return ''.join([w for w in text if ord(w) < 128])


def replace_maj_word(text: str) -> str:
    token = '<MAJ>'
    return ' '.join([w if not w.isupper() else token for w in delete_punctuation(text).split()])


def delete_digit(text: str) -> str:
    return re.sub('[0-9]+', '', text)


def first_line(text: str) -> str:
    return re.split(r'[.!?]', text)[0]


def last_line(text: str) -> str:
    if text.endswith('\n'): text = text[:-2]
    return re.split(r'[.!?]', text)[-1]


def delete_balise(text: str) -> str:
    return re.sub("<.*?>", "", text)


def stem(text: str) -> str:
    stemmer = EnglishStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text


def lemmatize(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text


def preprocess_text(text: str, args: Any, max_length: int=-1) -> str:
    if args.replace_maj_word_flag: text = replace_maj_word(text)
    if args.lower_flag: text = text.lower()
    if args.delete_punctuation_flag: text = delete_punctuation(text)
    if args.delete_balise_flag: text = delete_balise(text)
    if args.delete_stopwords_flag: text = delete_stopwords(text)
    if args.delete_non_ascii_flag: text = delete_non_ascii(text)
    if args.delete_digit_flag: text = delete_digit(text)
    if args.first_line_flag: text = first_line(text)
    if args.last_line_flag: text = last_line(text)
    if args.stem_flag: text = stem(text)
    if args.lemmatize_flag: text = lemmatize(text)
    if max_length > 0 and args.truncate_flag:
        text = str(text).strip().split()
        if len(text) > max_length:
            text = text[:max_length - 1] + ["..."]
        text = " ".join(text)
    return text
