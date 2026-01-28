import numpy as np


def track(matrix: np.ndarray, x: np.ndarray, turns: int) -> np.ndarray:
    coords = np.zeros((turns + 1, 4))
    coords[0, :] = np.copy(x)
    for i in range(turns):
        x = np.matmul(matrix, x)
        coords[i + 1, :] = np.copy(x)
    return coords