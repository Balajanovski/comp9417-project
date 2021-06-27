from scipy.sparse import lil_matrix, vstack
import pandas as pd
import numpy as np

def load_word_map():
    words = pd.read_csv("processed_data/word_map.csv", dtype="string", keep_default_na=False, na_filter=False)
    word_map = dict()
    for i, key in enumerate(words["0"]):
        word_map[key] = i
    return word_map

def load_train(word_map, words_to_vec):
    df = pd.read_csv("processed_data/processed_train.csv")
    X, y = df["question_text"], df["target"]
    X = vstack(list(map(lambda text: words_to_vec(str(text), word_map),X)))
    
    return X, y
'''
def load_test(word_map):
    df = pd.read_csv("processed_data/processed_train.csv")
    X, y = df["question_text"], df["target"]
    X = vstack(list(map(lambda text: words_to_vec(str(text), word_map),X)))
    
    return X, y, word_map

def words_to_vec(text, word_map):
    ret = lil_matrix((1,len(word_map)))
    if isinstance(text, float):
        print("float",text)
    for word in text.split(" "):
        x = word_map.get(word)
        if x:
            ret[:,x] = 1
    return ret
'''