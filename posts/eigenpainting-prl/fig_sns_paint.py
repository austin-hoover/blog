import argparse
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.stats
from tqdm import tqdm

from coupling import calc_eigvecs
from coupling import calc_eigtunes
from coupling import build_norm_matrix_from_tmat
from plot import CornerGrid
from utils import rotation_matrix
from utils import track

plt.style.use("style.mplstyle")


class Painter:
    def __init__(
        self,
        matrix: np.ndarray,
        turns: int,
        inj_size: int,
        inj_rms: float = 0.15,
        inj_cut: float = 3.0,
        method: str = "correlated",
    ) -> None:
        self.M = matrix
        self.V_inv = build_norm_matrix_from_tmat(self.M)
        self.V = np.linalg.inv(self.V_inv)

        self.eigvecs = calc_eigvecs(self.M)
        self.eigtunes = calc_eigtunes(self.M)
        
        self.inj_size = inj_size
        self.inj_rms = inj_rms
        self.inj_cut = np.repeat(inj_cut, 4)
        
        self.times = np.linspace(0.0, 1.0, turns + 1)

        self.method = method
        self.umax = None

    def set_inj_umax(self, umax: np.ndarray) -> None:
        self.umax = umax

    def set_inj_u(self, turn: int) -> np.ndarray:
        t = self.times[turn]
        if self.method == "correlated":
            return np.multiply(self.inj_umax, np.sqrt(t))
        elif self.method == "anti-correlated":
            tau1 = np.sqrt(1.0 - t)
            tau2 = np.sqrt(t)
            return np.multiply(self.inj_umax, [tau1, tau1, tau2, tau2])
        else:
            raise ValueError("Invalid method")

    def gen_bunch(self) -> np.ndarray:
        u = scipy.stats.truncnorm.rvs(
            scale=self.inj_rms,
            size=(self.n_inj, 4),
            a=-self.inj_cut,
            b=+self.inj_cut,
        )
        return u

    def paint(self, turn: list[int]) -> np.ndarray:
        bunches_n = [self.gen_bunch() for _ in range(nturns + 1)]
        
        for t in range(turns + 1):
            bunches_n[t] += self.get_inj_coords(t)

        for t, minipulse in enumerate(tqdm(bunches)):
            matrix = np.zeros((4, 4))
            matrix[0:2, 0:2] = rotation_matrix(2.0 * np.pi * self.eigtunes[0] * t)
            matrix[2:4, 2:4] = rotation_matrix(2.0 * np.pi * self.eigtunes[1] * t)
            bunches[t] = np.matmul(bunches[t], matrix.T)

        bunch = np.vstack(bunches)
        bunch = np.matmul(bunch. self.V.T)
        return bunch


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=2800)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--inj-size", type=int, default=200)
    parser.add_argument("--inj-rms", type=float, default=0.10)
    args = parser.parse_args()
    
    # Create output directory
    path = pathlib.Path(__file__)
    output_dir = os.path.join("outputs", path.stem)
    os.makedirs(output_dir, exist_ok=True)

    # Load transfer matrix
    M = np.loadtxt("data/matrix.txt")
    v1, v2 = calc_eigvecs(M)
    nu1, nu2 = calc_eigtunes(M)

    # Create painter
    painter = Painter(
        matrix=M,
        turns=args.turns,
        inj_size=args.inj_size,
        inj_rms=args.inj_rms,
    )

    # Run simulation
    turns_list = list(range(0, args.turns + args.stride, args.stride))
