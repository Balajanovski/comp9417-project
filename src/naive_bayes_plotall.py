import src.util as util
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from time import time
import tqdm
from src.metrics import print_metrics, get_metrics
import sys
import matplotlib.pyplot as plt
from typing import Dict

def run_naive_bayes(is_bernoulli: bool, min_ngram: int, max_ngram: int) -> Dict[str, float]:
    st = time()
    min_ngram = min_ngram if min_ngram != -1 else max_ngram
    X_train, X_test, y_train, y_test = util.load_data_bow(is_bernoulli, min_ngram, max_ngram)

    model = BernoulliNB() if is_bernoulli else MultinomialNB()

    print("Fitting model")
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)

    print(f"Time: {time()-st}s")
    
    return metrics

def plot_all(range_type) -> None:
    print("Starting plotall for naive bayes")
    n = 5
    start = 1 if range_type == "all" else -1
    data = dict()
    data["x_bernoulli"], data["y_bernoulli"] = [i for i in range(1,n+1)], [run_naive_bayes(True, start, i)["accuracy"] for i in range(1,n+1)]
    data["x_multinomial"], data["y_multinomial"]= [i for i in range(1,n+1)], [run_naive_bayes(False, start, i)["accuracy"] for i in range(1,n+1)]
    fig, ax = plt.subplots()
    for s in ["bernoulli", "multinomial"]:
        ax.plot(data[f"x_{s}"], data[f"y_{s}"], label=s)

    ax.legend()
    ax.set_xlabel("n-grams number")
    ax.set_ylabel("test accuracy")
    
    plt.show()

def main():
    args = sys.argv
    plot_all(args[1])

if __name__ == "__main__":
    main()
