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


class NormalizedPainter:
    def __init__(
        self,
        nu1: float,
        nu2: float,
        turns: int,
        inj_size: int,
        inj_rms: float = 0.15,
        inj_cut: float = 3.0,
        method: str = "correlated",
    ) -> None:
        self.nu1 = nu1
        self.nu2 = nu2
        self.inj_size = inj_size
        self.inj_rms = inj_rms
        self.inj_cut = np.repeat(inj_cut, 4)

        self.times = np.linspace(0.0, 1.0, turns + 1)

        self.method = method
        self.umax = None

    def set_umax(self, umax: np.ndarray) -> None:
        self.umax = np.array(umax)

    def get_inj_point(self, turn: int) -> np.ndarray:
        t = self.times[turn]
        if self.method == "corr":
            return np.multiply(self.umax, np.sqrt(t))
        elif self.method == "anticorr":
            tau1 = np.sqrt(1.0 - t)
            tau2 = np.sqrt(t)
            return np.multiply(self.umax, [tau1, tau1, tau2, tau2])
        else:
            raise ValueError("Invalid method")

    def gen_bunch(self) -> np.ndarray:
        return scipy.stats.truncnorm.rvs(
            scale=self.inj_rms,
            size=(self.inj_size, 4),
            a=-self.inj_cut,
            b=+self.inj_cut,
        )

    def phase_adv_matrix(self, turns: int) -> np.ndarray:
        mu1 = 2.0 * np.pi * self.nu1 * turns
        mu2 = 2.0 * np.pi * self.nu2 * turns
        matrix = np.zeros((4, 4))
        matrix[0:2, 0:2] = rotation_matrix(mu1)
        matrix[2:4, 2:4] = rotation_matrix(mu2)
        return matrix

    def paint(self, turns: list[int]) -> np.ndarray:
        bunches = [self.gen_bunch() for _ in range(turns + 1)]

        for t in range(turns + 1):
            bunches[t] += self.get_inj_point(t)

        for t, minipulse in enumerate(tqdm(bunches)):
            matrix = self.phase_adv_matrix(t)
            bunches[t] = np.matmul(bunches[t], matrix.T)

        return np.vstack(bunches)
