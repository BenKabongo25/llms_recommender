# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - July 2024

# Absa: Utils

# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Utils


import nltk
import numpy as np
import random
import re
import time
import torch

from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from typing import *


class AbsaData(object):

    SENTIMENT_PARAPHRASE_DEFAULT = {
        "very positive": "very great",
        "positive": "great",
        "neutral": "ok",
        "negative": "bad",
        "very negative": "very bad"
    }

    def __init__(
        self,
        aspects_categories: List[str]=[],
        aspects_terms: Optional[Dict[str, List[str]]]={},
        sentiment_polarities: Union[int, List[str]]=3,
        sentiment_paraphrases: Dict[str, str]={},
    ):
        self.aspects_categories = aspects_categories
        self.aspects_terms = aspects_terms

        self.sentiment_polarities = []
        if sentiment_paraphrases == {}:
            sentiment_paraphrases = AbsaData.SENTIMENT_PARAPHRASE_DEFAULT
        self.sentiment_paraphrases_kv = sentiment_paraphrases
        self.sentiment_paraphrases_vk = dict(
            list(map(lambda kv: (kv[1],kv[0]), self.sentiment_paraphrases_kv.items()))
        )
        self.set_sentiment_polarities(sentiment_polarities)


    def set_sentiment_polarities(self, sentiment_polarities: Union[int, List[str]]):
        if isinstance(sentiment_polarities, int):
            assert sentiment_polarities in {2, 3, 5}, "Invalid sentiment polarity value"
            if sentiment_polarities == 2:
                sentiment_polarities = ["positive", "negative"]
            elif sentiment_polarities == 3:
                sentiment_polarities = ["positive", "neutral", "negative"]
            elif sentiment_polarities == 5:
                sentiment_polarities = ["very positive", "positive", "neutral", "negative", "very negative"]
        
        self.sentiment_polarities = sentiment_polarities


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