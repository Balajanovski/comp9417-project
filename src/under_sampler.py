import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

def sample_train():
    df = pd.read_csv("data/train.csv")
    X, y = df["question_text"].to_numpy(), df["target"].to_numpy()

    sampler = RandomUnderSampler(random_state=0)
    X, y = sampler.fit_resample(X.reshape(-1,1),y)
    X = [row[0] for row in X]
    return X, y

def get_test():
    df = pd.read_csv("data/test.csv")
    X, y = df["question_text"].to_numpy(), df["target"].to_numpy()
    return X, y