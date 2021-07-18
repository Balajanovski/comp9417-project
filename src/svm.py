import src.util as util
from sklearn.svm import SVC, LinearSVC
from time import time
from src.metrics import print_metrics, get_metrics
import sys


def run_svm(path: str, type, kernel, c):
    st = time()
    if type == "bernoulli":
        X_train, X_test, y_train, y_test = util.load_data_bow(path, True, 1, 1)
    elif type == "multinomial":
        X_train, X_test, y_train, y_test = util.load_data_bow(path, False, 1, 1)
    elif type == "word2vec":
        X_train, X_test, y_train, y_test = util.load_data_word2vec_sentence(path)
    else:
        raise RuntimeError("invalid input, argument 1 must be either 'bernoulli', 'multinomial' or 'word2vec'")
    if kernel == "linear":
        # speed increase
        model = LinearSVC(verbose=1, C=c, class_weight={0:0.2,1:3}, max_iter=2000)
    else:
        model = SVC(kernel=kernel, verbose=1, C=c, class_weight="balanced")
    
    print("Fitting model")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)

    print(f"Time: {time()-st}s")


def main():
    args = sys.argv
    run_svm(args[4], args[1], args[2], int(args[3]))


if __name__ == "__main__":
    main()
