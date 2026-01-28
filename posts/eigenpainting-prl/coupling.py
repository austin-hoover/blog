import numpy as np


def build_poisson_matrix(ndim: int = 4) -> None:
    U = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        U[i : i + 2, i : i + 2] = [[0.0, 1.0], [-1.0, 0.0]]
    return U


def normalize_eigvec(v: np.ndarray) -> np.ndarray:
    ndim = v.shape[0]
    U = build_poisson_matrix(ndim)
    norm = np.abs(np.imag(np.linalg.multi_dot([np.conj(v), U, v])))
    if norm > 0:
        return v * np.sqrt(2.0 / norm)
    return v

    
def build_norm_matrix_from_eigvecs(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1 = normalize_eigvec(v1)
    v2 = normalize_eigvec(v2)
    
    V = np.zeros((4, 4))
    V[:, 0] = +v1.real
    V[:, 1] = -v1.imag
    V[:, 2] = +v2.real
    V[:, 3] = -v2.imag
    return np.linalg.inv(V)


def build_norm_matrix_from_tmat(M: np.ndarray) -> np.ndarray:
    eig_res = np.linalg.eig(M)
    v1 = normalize_eigvec(eig_res.eigenvectors[:, 0])
    v2 = normalize_eigvec(eig_res.eigenvectors[:, 2])
    return build_norm_matrix_from_eigvecs(v1, v2)


def calc_eigvecs(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eig_res = np.linalg.eig(M)
    v1 = normalize_eigvec(eig_res.eigenvectors[:, 0])
    v2 = normalize_eigvec(eig_res.eigenvectors[:, 2])
    return (v1, v2)


def calc_eigtunes(M: np.ndarray) -> tuple[float, float]:
    eig_res = np.linalg.eig(M)
    eigvals = eig_res.eigenvalues[::2]
    return np.arccos(np.real(eigvals)) / (2.0 * np.pi)
