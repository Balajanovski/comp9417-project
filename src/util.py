import pandas as pd
from processed_data.processed_data_folder import PROCESSED_DATA_FOLDER_PATH
import os
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from src.under_sampler import sample_data
from gensim.models import KeyedVectors
from embeddings import EMBEDDINGS_FOLDER_PATH

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

def load_data_raw() -> Tuple[List[str], np.array, List[str], np.array]:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, "processed_train.csv"))
    df["question_text"] = df["question_text"].apply(str)
    X, y = df["question_text"].to_list(), df["target"].to_numpy()
    X, y = sample_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

    return X_train, X_test, y_train, y_test

def load_data_bow(is_binary: bool = True, min_ngram: int = 1, max_ngram: int = 1) -> Tuple[np.array, np.array, np.array, np.array]:
    X_train, X_test, y_train, y_test = load_data_raw()

    X_train, X_test = create_bags_of_words(X_train, X_test, is_binary, min_ngram, max_ngram)

    return X_train, X_test, y_train, y_test

def load_data_word2vec_sentence() -> Tuple[np.array, np.array, np.array, np.array]:
    wordvec_map = KeyedVectors.load_word2vec_format(os.path.join(EMBEDDINGS_FOLDER_PATH, "GoogleNews-vectors-negative300", "GoogleNews-vectors-negative300.bin"), binary=True)
    X_train_strings, X_test_strings, y_train, y_test = load_data_raw()
    X_train = np.zeros((len(X_train_strings),300))
    X_test = np.zeros((len(X_test_strings),300))
    for i,s in enumerate(X_train_strings):
        X_train[i] = np.average([get_word2vec_from_map(word, wordvec_map) for word in s.split(" ")], axis=0)
    for i,s in enumerate(X_test_strings):
        X_test[i] = np.average([get_word2vec_from_map(word, wordvec_map) for word in s.split(" ")], axis=0)
    return X_train, X_test, y_train, y_test

def load_data_word2vec_deep_learning(sequence_length: Optional[int] = None) -> Tuple[np.array, np.array, np.array, np.array]:
    wordvec_map = KeyedVectors.load_word2vec_format(os.path.join(EMBEDDINGS_FOLDER_PATH, "GoogleNews-vectors-negative300", "GoogleNews-vectors-negative300.bin"), binary=True)
    X_train_strings, X_test_strings, y_train, y_test = load_data_raw()

    if sequence_length is None:
        sequence_length = max(map(lambda sentence: len(sentence.split(" ")), X_train_strings))

    X_train = np.zeros((len(X_train_strings), sequence_length, 300))
    X_test = np.zeros((len(X_test_strings), sequence_length, 300))

    for i, sentence in enumerate(X_train_strings):
        for j, word in enumerate(sentence.split(" ")):
            X_train[i][j] = get_word2vec_from_map(word, wordvec_map)
    for i, sentence in enumerate(X_test_strings):
        for j, word in enumerate(sentence.split(" ")):
            X_test[i][j] = get_word2vec_from_map(word, wordvec_map)

    return X_train, X_test, y_train, y_test

def get_word2vec_from_map(word: str, map) -> np.array:
    if not word in map:
        return np.zeros(300)
    return map[word]