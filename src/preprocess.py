import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import spacy
import numpy as np
from under_sampler import sample

if __name__ == "__main__":
    X, y = sample()

    nlp = spacy.load("en_core_web_sm")
    X = list(map(lambda text: " ".join([token.lemma_ for token in nlp(text) if not token.is_stop]), X))

    word_map = dict()
    count = 0
    for words in X:
        for word in words.split(" "):
            if not word in word_map:
                word_map[word] = count
                count += 1

    data_processed = {
        "question_text": X,
        "target": y,
    }

    word_map_list = [word for word in word_map.keys()]

    df_data = pd.DataFrame(data_processed)
    df_word_map = pd.DataFrame(word_map_list)

    df_data.to_csv("data/processed_train.csv", index=False)
    df_word_map.to_csv("data/word_map.csv", index=False)
    print(f"Number of different words: {count}")

'''
    def get_word_vector(word_list):
        vec = [0 for _ in range(count)]
        for word in word_list:
            index = word_map.get(word)
            if index:
                vec[index] += 1
        return vec
'''
    