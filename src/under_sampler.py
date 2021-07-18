import pandas as pd
from typing import Tuple, List
import numpy as np
import random


def sample_data(X: List[str], y: np.ndarray) -> Tuple[List[str], np.ndarray]:
    pos_index = []
    neg_index = []

    for i in range(y.shape[0]):
        if y[i]:
            pos_index.append(i)
        else:
            neg_index.append(i)

    print(
        f"Num positive examples: {len(pos_index)}. Num negative examples: {len(neg_index)}"
    )

    sample_amount = min(len(pos_index), len(neg_index))

    random.seed(8)
    final_index = random.sample(pos_index, sample_amount) + random.sample(
        neg_index, sample_amount
    )

    return [X[i] for i in final_index], y[final_index]
