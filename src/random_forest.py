from sklearn.utils import class_weight
import src.util as util
from sklearn.ensemble import RandomForestClassifier
from time import time
from src.metrics import print_metrics, get_metrics
import sys


def random_forest(path: str, max_depth, n_trees, type):
    st = time()
    if type == "bernoulli":
        X_train, X_test, y_train, y_test = util.load_data_bow(path, True, 1, 1)
    elif type == "multinomial":
        X_train, X_test, y_train, y_test = util.load_data_bow(path, False, 1, 1)
    elif type == "word2vec":
        X_train, X_test, y_train, y_test = util.load_data_word2vec_sentence_tfidf(path, "punct_stopwords_removed_lemmatized_word_map.csv", True)
    else:
        raise RuntimeError("third argument must be `bernoulli`, 'multinomial' or `word2vec`")

    model = RandomForestClassifier(
        n_estimators=n_trees, max_depth=max_depth, verbose=1, n_jobs=-1, class_weight="balanced"
    )

    print("Fitting model")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)

    print(f"Time: {time()-st}s")


def main():
    args = sys.argv
    max_depth = int(args[1])
    n_trees = int(args[2])
    type = args[3]
    path = args[4]

    random_forest(path, max_depth, n_trees, type)


if __name__ == "__main__":
    main()
