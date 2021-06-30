import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score


def print_metrics(y_pred: np.ndarray, y_actual: np.ndarray) -> None:
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    auc_roc = roc_auc_score(y_actual, y_pred)

    print(f"Metrics:\n"
          f"\tPrecision: {precision}\n"
          f"\tRecall: {recall}\n"
          f"\tAccuracy: {accuracy}\n"
          f"\tF1: {f1}\n"
          f"\tAuc roc: {auc_roc}\n")
