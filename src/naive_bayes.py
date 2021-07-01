import src.util as util
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from time import time
import tqdm
from src.print_metrics import print_metrics
import sys

def run_naive_bayes(is_bernouli: bool, min_ngram: int, max_ngram: int):
    st = time()

    X_train, X_test, y_train, y_test = util.load_data_bow(is_bernouli, min_ngram, max_ngram)

    model = BernoulliNB() if is_bernouli else MultinomialNB()

    print("Fitting model")
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    print_metrics(y_pred, y_test)

    print(f"Time: {time()-st}s")

def main():
    args = sys.argv

    if args[1] != "bernouli" and args[1] != "multinomial":
        raise "invalid input, argument 1 must be either 'bernouli' or 'multinomial'"
    
    print(f"Starting {args[1]}")
    run_naive_bayes(args[1] == "bernouli", int(args[2]), int(args[3]))

if __name__ == "__main__":
    main()
