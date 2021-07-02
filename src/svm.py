import src.util as util
from sklearn.svm import SVC, LinearSVC
from time import time
import tqdm
from src.metrics import print_metrics, get_metrics
import sys

def run_svm(is_bernouli, kernel):
    st = time()

    X_train, X_test, y_train, y_test = util.load_data_bow(is_bernouli, 1, 1)
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
    if args[1] != "bernouli" and args[1] != "multinomial":
        raise "invalid input, argument 1 must be either 'bernouli' or 'multinomial'"
    run_svm(args[1] == "bernouli", args[2])

if __name__ == "__main__":
    main()