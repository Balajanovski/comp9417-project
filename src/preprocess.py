import pandas as pd
import spacy
from src.under_sampler import sample_data
import numpy as np
from processed_data.processed_data_folder import PROCESSED_DATA_FOLDER_PATH
from data.data_folder import DATA_FOLDER_PATH
from sklearn.model_selection import train_test_split
import os
import tqdm
from typing import List


def main():
    df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "train.csv"))
    X_raw, y_raw = df["question_text"].to_list(), df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.3, random_state=8, stratify=y_raw)
    X_train, y_train = sample_data(X_train, y_train)

    X_train = nlp_cleanup(X_train)
    X_test = nlp_cleanup(X_test)

    save_data_csv(X_train, y_train, filename="processed_train.csv")
    save_data_csv(X_test, y_test, filename="processed_test.csv")
    create_and_save_word_map(X_train)


def nlp_cleanup(X: List[str]) -> np.ndarray:
    nlp = spacy.load("en_core_web_sm")

    cleaned_data = np.array([
        " ".join(token.lemma_ for token in nlp(X_row.lower()) if not token.is_stop and not token.is_punct)
        for X_row in tqdm.tqdm(X, desc="Cleaning data")
    ])

    return cleaned_data


def save_data_csv(X: np.ndarray, y: np.ndarray, filename: str) -> None:
    print("Saving data")

    data_processed = {
        "question_text": X,
        "target": y,
    }

    df = pd.DataFrame(data_processed)
    df.to_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, filename), index=False)

    print("Saved data")


def create_and_save_word_map(X: np.ndarray) -> None:
    word_map = {}

    count = 0
    for X_row in tqdm.tqdm(X, desc="Creating word map"):
        for word in X_row.split(" "):
            if not word in word_map:
                word_map[word] = count
                count += 1

    df_word_map = pd.DataFrame({"words": word_map.keys(), "encoded": word_map.values()})

    df_word_map.to_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, "word_map.csv"), index=False)
    print(f"Number of different words: {count}")


if __name__ == "__main__":
    main()
