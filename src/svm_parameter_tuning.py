import src.util as util
from sklearn.svm import SVC, LinearSVC
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV, KFold
from time import time
import tqdm
from src.metrics import print_metrics, get_metrics
import sys

class RandomClassWeight:
    def __init__(self, low, high):
        self.param = uniform(low, high)
    def rvs(self, random_state):
        return {0: self.param.rvs(), 1: self.param.rvs()}

def run_svm(path, type, kernel):
    st = time()
    if type == "bernoulli":
        X_train, X_test, y_train, y_test = util.load_data_bow(path, True, 1, 1)
    elif type == "multinomial":
        X_train, X_test, y_train, y_test = util.load_data_bow(path, False, 1, 1)
    elif type == "word2vec":
        X_train, X_test, y_train, y_test = util.load_data_word2vec_sentence_tfidf(path)
    else:
        raise "invalid input, argument 1 must be either 'bernoulli', 'multinomial' or 'word2vec'"
    # model = LinearSVC(verbose=1)
    if kernel == "linear":
        # speed increase
        model = LinearSVC(max_iter=2000)
    else:
        model = SVC(kernel=kernel, max_iter=2000)

    randomised_search = RandomizedSearchCV(
        model,
        cv=KFold(n_splits=4).split(X_train),
        param_distributions={"class_weight": RandomClassWeight(0, 100)},
        n_jobs=-1,
        n_iter=100,
        verbose=1,
        scoring="f1"
    )

    print("Fitting randomised search")
    randomised_search.fit(X_train, y_train)
    model = randomised_search.best_estimator_

    print(f"Best model: {model}")

    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)

    print(f"Time: {time()-st}s")
    print(randomised_search.cv_results_)

def main():
    args = sys.argv
    run_svm(args[3], args[1], args[2])


if __name__ == "__main__":
    main()
