import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from typing import Dict
import tensorflow.keras.metrics as metrics
from typing import List


def get_metrics(y_pred: np.ndarray, y_actual: np.ndarray) -> Dict[str, float]:
    metrics = dict()
    metrics["precision"] = precision_score(y_actual, y_pred)
    metrics["recall"] = recall_score(y_actual, y_pred)
    metrics["accuracy"] = accuracy_score(y_actual, y_pred)
    metrics["f1"] = f1_score(y_actual, y_pred)
    metrics["auc_roc"] = roc_auc_score(y_actual, y_pred)

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    print(
        f"Metrics:\n"
        f"\tPrecision: {metrics['precision']}\n"
        f"\tRecall: {metrics['recall']}\n"
        f"\tAccuracy: {metrics['accuracy']}\n"
        f"\tF1: {metrics['f1']}\n"
        f"\tAuc roc: {metrics['auc_roc']}\n"
    )


def make_keras_model_metrics() -> List[metrics.Metric]:
    return [
        metrics.TruePositives(name="tp"),
        metrics.FalsePositives(name="fp"),
        metrics.TrueNegatives(name="tn"),
        metrics.FalseNegatives(name="fn"),
        metrics.BinaryAccuracy(name="accuracy"),
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall"),
        metrics.AUC(name="aucpr", curve="PR"),
        metrics.AUC(name="aucroc", curve="ROC"),
    ]
