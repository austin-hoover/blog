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

    def set_umax(self, umax: np.ndarray) -> None:
        self.umax = np.array(umax)

    def get_inj_u(self, turn: int) -> np.ndarray:
        t = self.times[turn]
        if self.method == "corr":
            return np.multiply(self.umax, np.sqrt(t))
        elif self.method == "anticorr":
            tau1 = np.sqrt(1.0 - t)
            tau2 = np.sqrt(t)
            return np.multiply(self.umax, [tau1, tau1, tau2, tau2])
        else:
            raise ValueError("Invalid method")

    def get_inj_x(self, turn: int) -> np.ndarray:
        return np.matmul(self.V, self.get_inj_u(turn))

    def gen_bunch(self) -> np.ndarray:
        u = scipy.stats.truncnorm.rvs(
            scale=self.inj_rms,
            size=(self.inj_size, 4),
            a=-self.inj_cut,
            b=+self.inj_cut,
        )
        return u

    def paint(self, turns: list[int]) -> np.ndarray:
        bunches = [self.gen_bunch() for _ in range(turns + 1)]
        
        for t in range(turns + 1):
            bunches[t] += self.get_inj_u(t)

        for t, minipulse in enumerate(tqdm(bunches)):
            matrix = np.zeros((4, 4))
            matrix[0:2, 0:2] = rotation_matrix(2.0 * np.pi * self.eigtunes[0] * t)
            matrix[2:4, 2:4] = rotation_matrix(2.0 * np.pi * self.eigtunes[1] * t)
            bunches[t] = np.matmul(bunches[t], matrix.T)

        bunch = np.vstack(bunches)
        bunch = np.matmul(bunch, self.V.T)
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

    # Run simulations
    turns_list = list(range(0, args.turns + args.stride, args.stride))

    # Set max amplitude in normalized space.
    J1 = J2 = 25.0
    psi1 = np.pi * 0.00  # mode 1 phase
    psi2 = np.pi * 0.25  # mode 2 phase

    umax = np.zeros(4)
    umax[0] = np.sqrt(2.0 * J1) * np.cos(psi1)
    umax[1] = np.sqrt(2.0 * J1) * np.sin(psi1)
    umax[2] = np.sqrt(2.0 * J2) * np.cos(psi2)
    umax[3] = np.sqrt(2.0 * J2) * np.sin(psi2)
    painter.set_umax(umax)
    
    data = {}
    for method in ["corr", "anticorr", "flat", "eigen"]:
        data[method] = {}
        for key in ["bunch", "centroid"]:
            data[method][key] = []

    painter.method = "corr"
    painter.set_umax([1.0, 0.0, 1.0, 0.0])
    data["corr"]["bunch"] = [painter.paint(t) for t in turns_list]
    data["corr"]["centroid"] = [painter.get_inj_x(t) for t in turns_list]
