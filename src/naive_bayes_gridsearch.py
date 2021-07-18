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
    path: str, min_ngram: int, max_ngram: int
) -> Dict[str, float]:

    
    fig, ax = plt.subplots()

    for name, model in [("bernoulli", BernoulliNB()), ("multinomial", MultinomialNB())]:
        X_train, X_test, y_train, y_test = util.load_data_bow(
            path, name=="bernoulli", min_ngram, max_ngram
        )
        log_alpha = np.linspace(-3,2,50)
        alpha = 10 ** log_alpha
        param_search = GridSearchCV(model, verbose=1, cv=KFold(n_splits=4).split(X_train), n_jobs=-1, param_grid={"alpha": alpha}, scoring="f1")

        param_search.fit(X_train, y_train)
        model = param_search.best_estimator_
        print(name,model)

        y_pred = model.predict(X_test)
        metrics = get_metrics(y_pred, y_test)
        print_metrics(metrics)

        print(param_search.cv_results_)
        ax.plot(log_alpha, param_search.cv_results_["mean_test_score"],label=name)

    #plt.plot(log_alpha, param_search.cv_results_["mean_test_score"])
    ax.set_xlabel("log(alpha)")
    ax.set_ylabel("average CV test F1-score")
    ax.legend()
    plt.show()

    return metrics

def main():
    args = sys.argv

    run_naive_bayes(args[3], int(args[1]), int(args[2]))


if __name__ == "__main__":
    main()
