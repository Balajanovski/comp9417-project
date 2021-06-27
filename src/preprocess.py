import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import spacy
from under_sampler import sample_train, get_test

def process_save_csv(X, y, path):
    nlp = spacy.load("en_core_web_sm")
    X = list(map(lambda text: " ".join([token.lemma_ for token in nlp(text) if not token.is_stop]), X))

    data_processed = {
        "question_text": X,
        "target": y,
    }

    df = pd.DataFrame(data_processed)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    X_train, y_train = sample_train()
    #X_test, y_test = get_test()
    process_save_csv(X_train, y_train, "processed_data/processed_train.csv")
    #process_save_csv(X_test, y_test, "processed_data/processed_test.csv")

    nlp = spacy.load("en_core_web_sm")
    X = list(map(lambda text: " ".join([token.lemma_ for token in nlp(text) if not token.is_stop]), X_train))

    word_map = dict()
    count = 0
    for words in X:
        for word in words.split(" "):
            if not word in word_map:
                word_map[word] = count
                count += 1

    word_map_list = [word for word in word_map.keys()]

    df_word_map = pd.DataFrame(word_map_list)

    df_word_map.to_csv("data/word_map.csv", index=False)
    print(f"Number of different words: {count}")
    