import src.util as util
from sklearn.svm import SVC, LinearSVC
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
    if kernel == "linear":
        # speed increase
        model = LinearSVC(verbose=1)
    else:
        model = SVC(kernel=kernel, verbose=1)

    print("Fitting model")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)


    print(f"Time: {time()-st}s")

def main():
    args = sys.argv
    run_svm(args[1], args[2])

if __name__ == "__main__":
    main()