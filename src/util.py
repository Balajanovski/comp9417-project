from random import sample
import pandas as pd
from processed_data.processed_data_folder import PROCESSED_DATA_FOLDER_PATH
import os
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from src.under_sampler import sample_data

def load_word_map():
    word_map_df = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, "word_map.csv"), dtype="string", keep_default_na=False, na_filter=False)
    word_map = dict()

    for word, encoded in zip(word_map_df["words"], word_map_df["encoded"]):
        word_map[word] = int(encoded)
    
    return word_map

def create_bags_of_words(train_data: List[str], test_data: List[str], is_binary: bool, min_ngram: int, max_ngram: int) -> Tuple[np.array, np.array]:
    vectorizer = CountVectorizer(token_pattern=r"[^\s]+", binary=is_binary, ngram_range=(min_ngram, max_ngram))
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_test

def load_data(is_binary: bool = True, min_ngram: int = 1, max_ngram: int = 1) -> Tuple[np.array, np.array, np.array, np.array]:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, "processed_train.csv"))
    df["question_text"] = df["question_text"].apply(str)
    X, y = df["question_text"].to_list(), df["target"].to_numpy()
    X, y = sample_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

    X_train, X_test = create_bags_of_words(X_train, X_test, is_binary, min_ngram, max_ngram)

    return X_train, X_test, y_train, y_test
