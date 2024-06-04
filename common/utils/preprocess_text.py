# Ben Kabongo - MIA Paris-Saclay x Onepoint
# NLP & RecSys - June 2024

# Common
# Utils functions to preprocess text data

import nltk
import re
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from typing import *

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

def preprocess_text(text: str, args: Any) -> str:
    if args.lower_flag: text = text.lower()
    if args.delete_punctuation_flag: text = delete_punctuation(text)
    if args.delete_balise_flag: text = delete_balise(text)
    if args.delete_stopwords_flag: text = delete_stopwords(text)
    if args.delete_non_ascii_flag: text = delete_non_ascii(text)
    if args.replace_maj_word_flag: text = replace_maj_word(text)
    if args.delete_digit_flag: text = delete_digit(text)
    if args.first_line_flag: text = first_line(text)
    if args.last_line_flag: text = last_line(text)
    if args.stem_flag: text = stem(text)
    if args.lemmatize_flag: text = lemmatize(text)
    return text
