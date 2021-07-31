import src.util as util
from sklearn.svm import SVC, LinearSVC
from time import time
from src.metrics import print_metrics, get_metrics
import sys
import pickle

def run_pretrained_model(path: str, type, model_path):
    st = time()
    if type == "bernoulli":
        X_train, X_test, y_train, y_test = util.load_data_bow(path, True, 1, 1)
    elif type == "multinomial":
        X_train, X_test, y_train, y_test = util.load_data_bow(path, False, 1, 1)
    elif type == "word2vec":
        X_train, X_test, y_train, y_test = util.load_data_word2vec_sentence_tfidf(path)
    else:
        raise RuntimeError("invalid input, argument 1 must be either 'bernoulli', 'multinomial' or 'word2vec'")
    model = util.load_model(model_path)
    
    print("Running model on test set")
    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)
    
    print(f"Time: {time()-st}s")


def main():
    args = sys.argv
    run_pretrained_model(args[3], args[1], args[2])


if __name__ == "__main__":
    main()
