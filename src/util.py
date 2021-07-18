import pandas as pd
from processed_data.processed_data_folder import PROCESSED_DATA_FOLDER_PATH
import os
from typing import List, Tuple, Optional, Dict, Generator
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from gensim.models import KeyedVectors
from embeddings import EMBEDDINGS_FOLDER_PATH
import matplotlib.pyplot as plt
from plots import PLOTS_FOLDER_PATH
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from src.under_sampler import sample_data


def create_bags_of_words(
    train_data: List[str],
    test_data: List[str],
    is_binary: bool,
    min_ngram: int,
    max_ngram: int,
) -> Tuple[np.array, np.array]:
    vectorizer = CountVectorizer(
        token_pattern=r"[^\s]+", binary=is_binary, ngram_range=(min_ngram, max_ngram)
    )

    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_test


def load_data_raw(path: str, portion_to_load: float = 1.0) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, path)).replace(np.nan, '', regex=True)
    X_raw, y_raw = df["question_text"].to_list(), df["target"].to_numpy()

    portion_indices = np.random.randint(0, len(X_raw), size=(round(len(X_raw) * portion_to_load),))
    portion_X = [X_raw[i] for i in portion_indices]
    portion_Y = y_raw[portion_indices]

    return train_test_split(portion_X, portion_Y, test_size=0.3, random_state=42, stratify=portion_Y)


def load_data_bow(
    path: str, is_binary: bool = True, min_ngram: int = 1, max_ngram: int = 1
) -> Tuple[np.array, np.array, np.array, np.array]:
    X_train, X_test, y_train, y_test = load_data_raw(path)

    X_train, X_test = create_bags_of_words(
        X_train, X_test, is_binary, min_ngram, max_ngram
    )

    return X_train, X_test, y_train, y_test


def load_data_word2vec_sentence(
    path: str,
) -> Tuple[np.array, np.array, np.array, np.array]:
    wordvec_map = KeyedVectors.load_word2vec_format(
        os.path.join(
            EMBEDDINGS_FOLDER_PATH,
            "GoogleNews-vectors-negative300",
            "GoogleNews-vectors-negative300.bin",
        ),
        binary=True,
    )
    X_train_strings, X_test_strings, y_train, y_test = load_data_raw(path)
    X_train = np.zeros((len(X_train_strings), 300))
    X_test = np.zeros((len(X_test_strings), 300))
    for i, s in enumerate(X_train_strings):
        X_train[i] = np.average(
            [get_word2vec_from_map(word, wordvec_map) for word in s.split(" ")], axis=0
        )
    for i, s in enumerate(X_test_strings):
        X_test[i] = np.average(
            [get_word2vec_from_map(word, wordvec_map) for word in s.split(" ")], axis=0
        )
    return X_train, X_test, y_train, y_test


def load_data_word2vec_deep_learning(
    path: str, sequence_length: Optional[int] = None, portion_to_load: float = 1.0, balance: bool = False, batch_size: int = 32, validation_split: float = 0.2
) -> Tuple[Generator, Generator, Generator, np.ndarray, np.ndarray, np.ndarray]:
    wordvec_map = KeyedVectors.load_word2vec_format(
        os.path.join(
            EMBEDDINGS_FOLDER_PATH,
            "GoogleNews-vectors-negative300",
            "GoogleNews-vectors-negative300.bin",
        ),
        binary=True,
    )
    word_vec_dims = get_word2vec_from_map("the", wordvec_map).shape[0]
    print(f"Word 2 vec dimensions {word_vec_dims}")

    X_train_strings, X_test_strings, y_train, y_test = load_data_raw(path, portion_to_load=portion_to_load)
    X_train_strings, X_val_strings, y_train, y_val = train_test_split(X_train_strings, y_train, test_size=0.2, stratify=y_train)

    if balance:
        X_train_strings, y_train = sample_data(X_train_strings, y_train)

    if sequence_length is None:
        sequence_length = max(
            map(lambda sentence: len(sentence.split(" ")), X_train_strings)
        )

    def _create_generator(X, y):
        batch_X = np.zeros((batch_size, sequence_length, word_vec_dims))
        batch_y = np.zeros((batch_size))
        batch_i = 0

        while True:
            for i in range(len(X_train_strings)):
                if batch_i >= batch_size:
                    yield batch_X, batch_y

                    batch_X = np.zeros(batch_size, sequence_length, word_vec_dims)
                    batch_y = np.zeros(batch_size)
                    batch_i = 0

                for j, word in enumerate(X_train_strings[i].split(" ")):
                    if j >= sequence_length:
                        break
                    batch_X[batch_i][j] = get_word2vec_from_map(word, wordvec_map)

                batch_y[batch_i] = y_train[i]
                batch_i += 1

    return _create_generator(X_train_strings, y_train), _create_generator(X_val_strings, y_val), _create_generator(X_test_strings, y_test), y_train, y_val, y_test


def get_word2vec_from_map(word: str, map) -> np.array:
    if not word in map:
        return np.zeros(300)
    return map[word]


def plot_keras_model_learning_curves(history, prefix: str) -> None:
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(os.path.join(PLOTS_FOLDER_PATH, f"{prefix}_accuracy_curve.png"))
    plt.clf()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(os.path.join(PLOTS_FOLDER_PATH, f"{prefix}_loss_curve.png"))
    plt.clf()


def make_class_weights(labels: np.ndarray) -> Dict:
    unique_labels = np.unique(labels)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=unique_labels,
                                                      y=labels)
    return {label: weight for label, weight in zip(unique_labels, class_weights)}
