import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple
import numpy as np
from data.data_folder import DATA_FOLDER_PATH
import os


def sample_raw_data() -> Tuple[np.ndarray, np.ndarray]:
    print("Sampling raw data")

    df = pd.read_csv(os.path.join(DATA_FOLDER_PATH, "train.csv"))
    X, y = df["question_text"].to_numpy(), df["target"].to_numpy()

    sampler = RandomUnderSampler(random_state=0)
    X, y = sampler.fit_resample(X.reshape(-1,1),y)

    print("Sampled raw data")

    return X, y
