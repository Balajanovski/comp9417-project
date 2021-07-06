import src.util as util
from sklearn.ensemble import RandomForestClassifier
from time import time
import tqdm
from src.metrics import print_metrics, get_metrics
import sys

def random_forest(max_depth, n_trees):
    st = time()

    X_train, X_test, y_train, y_test = util.load_data_bow(True, 1, 1)
    
    model = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, verbose=1, n_jobs=-1)

    print("Fitting model")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)


    print(f"Time: {time()-st}s")

def main():
    args = sys.argv
    max_depth = int(args[1])
    n_trees = int(args[2])

    random_forest(max_depth, n_trees)

if __name__ == "__main__":
    main()