import src.util as util
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from time import time
from src.metrics import print_metrics, get_metrics
import sys
import matplotlib.pyplot as plt
from typing import Dict
from plots import PLOTS_FOLDER_PATH
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
import os


def run_naive_bayes(
    path: str, is_bernoulli: bool, min_ngram: int, max_ngram: int
) -> Dict[str, float]:
    st = time()

    X_train, X_test, y_train, y_test = util.load_data_bow(
        path, is_bernoulli, min_ngram, max_ngram
    )

    model = BernoulliNB() if is_bernoulli else MultinomialNB()
    log_alpha = np.linspace(-3,2,50)
    alpha = 10 ** log_alpha
    param_search = GridSearchCV(model, verbose=1, cv=KFold(n_splits=4).split(X_train), n_jobs=-1, param_grid={"alpha": alpha}, scoring="f1")

    print("Fitting grid search")
    param_search.fit(X_train, y_train)
    model = param_search.best_estimator_
    print(model)

    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)

    print(f"Time: {time()-st}s")

    print(param_search.cv_results_)

    plt.plot(log_alpha, param_search.cv_results_["mean_test_score"])
    plt.xlabel("log(alpha)")
    plt.ylabel("average CV test F1-score")
    plt.show()

    return metrics

def main():
    args = sys.argv

    if args[1] != "bernoulli" and args[1] != "multinomial":
        raise RuntimeError(
            "invalid input, argument 1 must be either 'bernoulli' or 'multinomial'"
        )
    run_naive_bayes(args[4], args[1] == "bernoulli", int(args[2]), int(args[3]))


if __name__ == "__main__":
    main()
