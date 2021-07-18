import pandas as pd
import spacy
import numpy as np
from processed_data.processed_data_folder import PROCESSED_DATA_FOLDER_PATH
from data.data_folder import DATA_FOLDER_PATH
import os
import tqdm
from typing import List
import re


def main():
    df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "train.csv"))
    X_raw, y_raw = df["question_text"].to_list(), df["target"].to_numpy()

    docs = list(make_spacy_docs(X_raw))

    for processing, prefix in [
        (punctuation_removed(docs), "punct_removed"),
        (punct_removed_and_stopwords_removed(docs), "punct_stopwords_removed"),
        (punct_removed_lemmatized(docs), "punct_removed_lemmatized"),
        (
            punct_stopwords_removed_lemmatized(docs),
            "punct_stopwords_removed_lemmatized",
        ),
    ]:
        save_data_csv(processing, y_raw, filename=f"{prefix}.csv")
        create_and_save_word_map(processing, filename=f"{prefix}_word_map.csv")


def make_spacy_docs(X: List[str]):
    clean_non_ascii = re.compile(r"[^\x00-\x7f]")

    nlp = spacy.load("en_core_web_sm")
    docs = nlp.pipe(
        [clean_non_ascii.sub(" ", X_row.lower()) for X_row in X],
        n_process=8,
    )

    return docs


def punctuation_removed(docs) -> List[str]:
    return [
        " ".join(token.orth_ for token in doc if not token.is_punct) for doc in docs
    ]


def punct_removed_and_stopwords_removed(docs) -> List[str]:
    return [
        " ".join(
            token.orth_ for token in doc if not token.is_punct and not token.is_stop
        )
        for doc in docs
    ]


def punct_removed_lemmatized(docs) -> List[str]:
    return [
        " ".join(token.lemma_ for token in doc if not token.is_punct) for doc in docs
    ]


def punct_stopwords_removed_lemmatized(docs) -> List[str]:
    return [
        " ".join(
            token.lemma_ for token in doc if not token.is_punct and not token.is_stop
        )
        for doc in docs
    ]


def save_data_csv(X: List[str], y: np.ndarray, filename: str) -> None:
    print("Saving data")

    data_processed = {
        "question_text": X,
        "target": y,
    }

    df = pd.DataFrame(data_processed)
    df.to_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, filename), index=False)

    print("Saved data")


def create_and_save_word_map(X: List[str], filename: str) -> None:
    word_map = {}

    count = 0
    for X_row in tqdm.tqdm(X, desc="Creating word map"):
        for word in X_row.split(" "):
            if not word in word_map:
                word_map[word] = count
                count += 1

    df_word_map = pd.DataFrame({"words": word_map.keys(), "encoded": word_map.values()})

    df_word_map.to_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, filename), index=False)
    print(f"Number of different words: {count}")


if __name__ == "__main__":
    main()
