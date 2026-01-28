import numpy as np


def build_poisson_matrix(ndim: int = 4) -> None:
    U = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        U[i : i + 2, i : i + 2] = [[0.0, 1.0], [-1.0, 0.0]]
    return U


def normalize_eigvec(v: np.ndarray) -> np.ndarray:
    U = build_poisson_matrix(len(v))

    def norm(v):
        return np.linalg.multi_dot([np.conj(v), U, v])

    return v * np.sqrt(2.0 / np.abs(norm(v)))

    
def calc_eigvecs(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eig_res = np.linalg.eig(M)
    v1 = normalize_eigvec(eig_res.eigenvectors[:, 0])
    v2 = normalize_eigvec(eig_res.eigenvectors[:, 2])
    return (v1, v2)


def calc_eigtunes(M: np.ndarray) -> tuple[float, float]:
    eig_res = np.linalg.eig(M)
    eigvals = eig_res.eigenvalues[::2]
    return np.arccos(np.real(eigvals)) / (2.0 * np.pi)


def build_norm_matrix_from_eigvecs(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    V = np.zeros((4, 4))
    V[:, 0] = +np.real(v1)
    V[:, 1] = -np.imag(v1)
    V[:, 2] = +np.real(v2)
    V[:, 3] = -np.imag(v2)
    return np.linalg.inv(V)


def build_norm_matrix_from_tmat(M: np.ndarray) -> np.ndarray:
    v1, v2 = calc_eigvecs(M)
    return build_norm_matrix_from_eigvecs(v1, v2)
