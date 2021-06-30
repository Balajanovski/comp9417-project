import pandas as pd
import spacy
from src.under_sampler import sample_raw_data
import numpy as np
from processed_data.processed_data_folder import PROCESSED_DATA_FOLDER_PATH
import os
import tqdm


def main():
    X, y = sample_raw_data()

    X = nlp_cleanup(X)
    save_data_csv(X, y)
    create_and_save_word_map(X)


def nlp_cleanup(X: np.ndarray) -> np.ndarray:
    nlp = spacy.load("en_core_web_sm")

    cleaned_data = np.array([
        " ".join(token.lemma_ for token in nlp(X_row[0].lower()) if not token.is_stop and token.lemma_ in nlp.vocab)
        for X_row in tqdm.tqdm(X, desc="Cleaning data")
    ])

    return cleaned_data


def save_data_csv(X: np.ndarray, y: np.ndarray) -> None:
    print("Saving data")

    data_processed = {
        "question_text": X,
        "target": y,
    }

    df = pd.DataFrame(data_processed)
    df.to_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, "processed_train.csv"), index=False)

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
