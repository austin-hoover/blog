import numpy as np


def rotation_matrix(angle: float) -> np.ndarray:
    matrix = np.zeros((2, 2))
    matrix[0, 0] = +np.cos(angle)
    matrix[1, 1] = +np.cos(angle)
    matrix[0, 1] = +np.sin(angle)
    matrix[1, 0] = -np.sin(angle)
    return matrix


def track(matrix: np.ndarray, x: np.ndarray, turns: int) -> np.ndarray:
    coords = np.zeros((turns + 1, 4))
    coords[0, :] = np.copy(x)
    for i in range(turns):
        x = np.matmul(matrix, x)
        coords[i + 1, :] = np.copy(x)
    return coords
