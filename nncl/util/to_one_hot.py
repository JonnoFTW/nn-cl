import numpy as np
from pyopencl import cltypes


def to_one_hot(arr: np.ndarray) -> np.ndarray:
    b = np.max(arr) + 1
    return np.eye(b)[arr].astype(cltypes.float)