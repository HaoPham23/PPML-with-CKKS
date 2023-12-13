import base64
import numpy as np
import pandas as pd
import random

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def write_data(file_name: str, data: bytes) -> int:
    data = base64.b64encode(data)
    with open(file_name, 'wb') as f: 
        f.write(data)
    return len(data)

def read_data(file_name: str) -> bytes:
    with open(file_name, 'rb') as f:
        data = f.read()
    return base64.b64decode(data)

def to_base64(data: bytes) -> bytes:
    return base64.b64encode(data)

def from_base64(data: bytes) -> bytes:
    return base64.b64decode(data)

def to_megabytes(size_in_bytes: int) -> float:
    return round(size_in_bytes/2**20, 2)

def pick_a_random_data_from_test_set():
    data = pd.read_csv("../../data/framingham.csv")
    X = data.drop(['TenYearCHD'], axis=1, inplace=False)
    Y = np.array(data['TenYearCHD'])
    X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
    # Standardize data
    X = np.array((X - X.mean()) / X.std())
    idx = random.choice(range(len(X)))
    return idx, X[idx], Y[idx]