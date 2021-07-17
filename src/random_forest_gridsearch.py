import src.util as util
from sklearn.ensemble import RandomForestClassifier
from time import time
import tqdm
from src.metrics import print_metrics, get_metrics
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold

def random_forest(n_trees, type):
    st = time()
    if type == "bernoulli":
        X_train, X_test, y_train, y_test = util.load_data_bow(True, 1, 1)
    elif type == "word2vec":
        X_train, X_test, y_train, y_test = util.load_data_word2vec_sentence()
    else:
        raise "third argument must be `bernouli` or `word2vec`"
    
    max_depth = np.linspace(2, 4, 10)
    max_depth = 10 ** max_depth

    model = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1)
    param_search = GridSearchCV(model, verbose=1, cv=KFold(n_splits=4).split(X_train), n_jobs=-1, param_grid={"max_depth": max_depth})

    print("Fitting parameter search for random forest")
    param_search.fit(X_train, y_train)
    model = param_search.best_estimator_
    print(f"Best model: {model}")
    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)


    print(f"Time: {time()-st}s")

def main():
    args = sys.argv
    n_trees = int(args[1])
    type = args[2]

    random_forest(n_trees, type)

if __name__ == "__main__":
    main()