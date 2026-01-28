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
    V = build_norm_matrix_from_tmat(M)
    v1, v2 = calc_eigvecs(M)
    nu1, nu2 = calc_eigtunes(M)

    # Create painter
    painter = NormalizedPainter(
        nu1=nu1,
        nu2=nu2,
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
    data["corr"]["bunch_n"] = [painter.paint(t) for t in turns_list]
    data["corr"]["inj_n"] = [painter.get_inj_point(t) for t in turns_list]

    data["corr"]["bunch"] = [np.matmul(bunch, V.T) for bunch in data["corr"]["bunch_n"]]
    data["corr"]["inj"] = [np.matmul(V, u) for u in data["corr"]["inj_n"]]
