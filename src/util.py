import pandas as pd
from processed_data.processed_data_folder import PROCESSED_DATA_FOLDER_PATH
import os


def load_word_map():
    word_map_df = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, "word_map.csv"), dtype="string", keep_default_na=False, na_filter=False)
    word_map = dict()
    for word, encoded in zip(word_map_df["words"], word_map_df["encoded"]):
        word_map[word] = int(encoded)
    return word_map


def load_train():
    df = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER_PATH, "processed_train.csv"))
    return df["question_text"], df["target"]
