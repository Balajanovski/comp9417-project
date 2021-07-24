import src.util as util
from sklearn.ensemble import RandomForestClassifier
from time import time
import tqdm
from src.metrics import print_metrics, get_metrics
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt

def random_forest(path: str, n_trees: int, type: str):
    st = time()
    if type == "bernoulli":
        X_train, X_test, y_train, y_test = util.load_data_bow(path, True, 1, 1)
    elif type == "multinomial":
        X_train, X_test, y_train, y_test = util.load_data_bow(path, False, 1, 1)
    elif type == "word2vec":
        X_train, X_test, y_train, y_test = util.load_data_word2vec_sentence(path)
    else:
        raise "third argument must be `bernoulli` or `word2vec`"
    
    max_depth_log = np.linspace(2, 4, 20)
    max_depth = 10 ** max_depth_log

    model = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=8)
    param_search = GridSearchCV(model, verbose=1, cv=KFold(n_splits=4).split(X_train), n_jobs=-1, param_grid={"max_depth": max_depth}, scoring=("f1","accuracy"), refit="f1")

    print("Fitting parameter search for random forest")
    param_search.fit(X_train, y_train)
    model = param_search.best_estimator_
    print(f"Best model: {model}")
    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)

    print(f"Time: {time()-st}s")

    plt.plot(max_depth_log, param_search.cv_results_["mean_test_f1"])
    plt.xlabel("log(max_depth)")
    plt.ylabel("average CV test F1-score")
    plt.savefig(f"plots/random_forest_search_{type}.png")
    print(param_search.cv_results_)

def main():
    args = sys.argv
    random_forest(args[3], int(args[1]), args[2])

if __name__ == "__main__":
    main()