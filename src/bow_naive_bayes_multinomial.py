import src.util as util
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from time import time
import numpy as np
from scipy.sparse import lil_matrix
from typing import Dict, List
import tqdm
from src.print_metrics import print_metrics


def main():
    st = time()

    word_map = util.load_word_map()
    X, y = util.load_train()
    X = create_bags_of_words(X, word_map)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
    model = MultinomialNB()

    print("Fitting model")
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    print_metrics(y_pred, y_test)

    print(f"Time: {time()-st}s")


def create_bags_of_words(texts: List[str], word_map: Dict[str, int]) -> lil_matrix:
    ret = lil_matrix((len(texts), len(word_map)))

    for i, text in enumerate(tqdm.tqdm(texts, desc="Creating bag of words")):
        if isinstance(text, float):
            continue

        for word in text.split(" "):
            ret[i, word_map[word]] += 1

    return ret


if __name__ == "__main__":
    main()
