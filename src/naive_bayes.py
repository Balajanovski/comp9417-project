import src.util as util
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from time import time
from src.metrics import print_metrics, get_metrics
import sys
import matplotlib.pyplot as plt
from typing import Dict

def run_naive_bayes(is_bernoulli: bool, min_ngram: int, max_ngram: int) -> Dict[str, float]:
    st = time()

    X_train, X_test, y_train, y_test = util.load_data_bow(is_bernoulli, min_ngram, max_ngram)

    model = BernoulliNB() if is_bernoulli else MultinomialNB()

    print("Fitting model")
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)

    print(f"Time: {time()-st}s")
    
    return metrics

def plot_all(n: int, is_bernoulli: bool) -> None:
    print(f"Starting {'bernoulli' if is_bernoulli else 'multinomial'} for ngrams up to {n}")

    x, y = [i for i in range(1,n+1)], [run_naive_bayes(is_bernoulli, 1, i)["accuracy"] for i in range(1,n+1)]
    plt.xlabel("n-grams number")
    plt.ylabel("accuracy")
    plt.plot(x,y)
    plt.show()

def main():
    args = sys.argv

    if args[1] != "bernoulli" and args[1] != "multinomial":
        raise "invalid input, argument 1 must be either 'bernoulli' or 'multinomial'"
    
    if len(args) == 3:
        plot_all(int(args[2]), args[1] == "bernoulli")
    else:
        run_naive_bayes(args[1] == "bernoulli", int(args[2]), int(args[3]))

if __name__ == "__main__":
    main()
