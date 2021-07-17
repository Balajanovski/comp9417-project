import src.util as util
from sklearn.svm import SVC, LinearSVC
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV, KFold
from time import time
import tqdm
from src.metrics import print_metrics, get_metrics
import sys

def run_svm(type, kernel):
    st = time()
    if type == "bernoulli":
        X_train, X_test, y_train, y_test = util.load_data_bow(True, 1, 1)
    elif type == "multinomial":
        X_train, X_test, y_train, y_test = util.load_data_bow(False, 1, 1)
    elif type == "word2vec":
        X_train, X_test, y_train, y_test = util.load_data_word2vec_sentence()
    else:
        raise "invalid input, argument 1 must be either 'bernoulli', 'multinomial' or 'word2vec'"
    #model = LinearSVC(verbose=1)
    if kernel == "linear_l2":
        # speed increase
        model = LinearSVC(penalty="l2")
    else:
        model = SVC(kernel=kernel)

    randomised_search = RandomizedSearchCV(model, cv = KFold(n_splits=4).split(X_train), param_distributions={"C":uniform(0.001, 2)}, random_state=8, n_jobs=-1, n_iter=12, verbose=1)

    print("Fitting randomised search")
    randomised_search.fit(X_train, y_train)
    model = randomised_search.best_estimator_

    print(f"Best model: {model}")

    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)

    print(f"Time: {time()-st}s")

def main():
    args = sys.argv
    run_svm(args[1], args[2])

if __name__ == "__main__":
    main()