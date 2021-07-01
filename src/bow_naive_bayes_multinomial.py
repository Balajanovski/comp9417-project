from os import error
import src.util as util
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from time import time
import numpy as np
from scipy.sparse import lil_matrix
from typing import Dict, List
import tqdm
from src.print_metrics import print_metrics
from sklearn.feature_extraction.text import CountVectorizer
import sys

def run_naive_bayes(is_bernouli: bool, min_ngram: int, max_ngram: int):
    st = time()

    word_map = util.load_word_map()
    X, y = util.load_train()
    X = create_bags_of_words(X, word_map, is_bernouli, min_ngram, max_ngram)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
    model = MultinomialNB()

    print("Fitting model")
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    print_metrics(y_pred, y_test)

    print(f"Time: {time()-st}s")


def create_bags_of_words(texts: List[str], word_map: Dict[str, int], is_bernouli: bool, min_ngram: int, max_ngram: int) -> lil_matrix:
    vectorizer = CountVectorizer(vocabulary=word_map, token_pattern=r"[^\s]+", binary=is_bernouli, ngram_range=(min_ngram, max_ngram))
    ret = vectorizer.fit_transform(texts)

    return ret

def main():
    args = sys.argv

    if args[1] != "bernouli" and args[1] != "multinomial":
        raise "invalid input, argument 1 must be either 'bernouli' or 'multinomial'"
    
    print(f"Starting {args[1]}")
    run_naive_bayes(args[1] == "bernouli", int(args[2]), int(args[3]))

if __name__ == "__main__":
    main()
