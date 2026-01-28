import argparse
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.stats
from tqdm import tqdm

from plot import CornerGrid
from utils import rotation_matrix
from utils import track

plt.style.use("style.mplstyle")


class Painter:
    def __init__(
        self,
        tune_x: float,
        tune_y: float,
        n_turns: int,
        n_inj: int,
        inj_rms: float = 0.15,
        inj_cut: float = 3.0,
        method: str = "correlated",
    ) -> None:
        self.tune_x = tune_x
        self.tune_y = tune_y
        self.n_inj = n_inj
        self.n_turns = n_turns
        self.inj_rms = inj_rms
        self.inj_cut = np.repeat(inj_cut, 4)
        self.times = np.linspace(0.0, 1.0, n_turns + 1)

        self.method = method
        self.xmax = None
        self.is_initialized = False

    def set_inj_xmax(self, xmax: np.ndarray) -> None:
        self.xmax = xmax

    def get_inj_coords(self, turn: int) -> np.ndarray:
        t = self.times[turn]
        if self.method == "correlated":
            return np.multiply(self.inj_xmax, np.sqrt(t))
        elif self.method == "anti-correlated":
            tau1 = np.sqrt(1.0 - t)
            tau2 = np.sqrt(t)
            return np.multiply(self.inj_xmax, [tau1, tau1, tau2, tau2])
        else:
            raise ValueError("Invalid method")

    def gen_bunch(self) -> np.ndarray:
        x = scipy.stats.truncnorm.rvs(
            scale=self.inj_rms,
            size=(self.n_inj, 4),
            a=-self.inj_cut,
            b=+self.inj_cut,
        )
        return x

    def paint(self, nturns: list[int]) -> np.ndarray:
        # Generate `n_turns` minipulses at the origin.
        bunches = [self.gen_bunch() for _ in range(nturns + 1)]

        # Move each minipulse to its final amplitude.
        for t in range(nturns + 1):
            bunches[t] += self.get_inj_coords(t)

        # Rotate each minipulse by the requested number of turns.
        for t, minipulse in enumerate(tqdm(bunches)):
            matrix = np.zeros((4, 4))
            matrix[0:2, 0:2] = rotation_matrix(2.0 * np.pi * self.tune_x * t)
            matrix[2:4, 2:4] = rotation_matrix(2.0 * np.pi * self.tune_y * t)
            bunches[t] = np.matmul(bunches[t], matrix.T)

        return np.vstack(bunches)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nturns", type=int, default=2800)
    parser.add_argument("--nparts", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--inj-rms", type=float, default=0.10)
    args = parser.parse_args()

    # Create output directory
    # ----------------------------------------------------------------------
    path = pathlib.Path(__file__)
    output_dir = os.path.join("outputs", path.stem)
    os.makedirs(output_dir, exist_ok=True)

    # Setup painter
    # ----------------------------------------------------------------------
    tune_x = 0.1810201  # horizontal tune
    tune_y = tune_x - 0.143561  # vertical tune

    turns_list = list(range(0, args.nturns + args.stride, args.stride))

    # Create painter
    # ----------------------------------------------------------------------
    painter = Painter(
        tune_x=tune_x,
        tune_y=tune_y,
        n_turns=args.nturns,
        n_inj=args.nparts,
        inj_rms=args.inj_rms,
    )

    # Run simulations
    # ----------------------------------------------------------------------
    data = {}
    for method in ["corr", "anticorr", "flat", "eig"]:
        data[method] = {}
        for key in ["bunch", "centroid"]:
            data[method][key] = []

    painter.method = "anti-correlated"
    painter.inj_xmax = np.array([1.0, 0.0, 1.0, 0.0])
    data["anticorr"]["bunch"] = [painter.paint(t) for t in turns_list]
    data["anticorr"]["centroid"] = [painter.get_inj_coords(t) for t in turns_list]

    painter.method = "correlated"
    painter.inj_xmax = np.array([1.0, 0.0, 1.0, 0.0])
    data["corr"]["bunch"] = [painter.paint(t) for t in turns_list]
    data["corr"]["centroid"] = [painter.get_inj_coords(t) for t in turns_list]

    painter.method = "correlated"
    painter.inj_xmax = np.array([1.0, 0.0, 0.0, 0.0])
    data["flat"]["bunch"] = [painter.paint(t) for t in turns_list]
    data["flat"]["centroid"] = [painter.get_inj_coords(t) for t in turns_list]

    painter.method = "correlated"
    painter.inj_xmax = np.array([1.0, 0.0, 0.0, 1.0])  # eigenvector
    painter.tune_x = tune_x
    painter.tune_y = tune_x
    data["eig"]["bunch"] = [painter.paint(t) for t in turns_list]
    data["eig"]["centroid"] = [painter.get_inj_coords(t) for t in turns_list]

    # Make plots
    # ----------------------------------------------------------------------
    for method in data:
        output_subdir = os.path.join(output_dir, method)
        os.makedirs(output_subdir, exist_ok=True)

        # Settings
        limits = 4 * [(-2.0, 2.0)]
        labels = ["x", "x'", "y", "y'"]
        bins = 64
        cmap = "gray_r"
        blur = 1.0

        # Global scaling
        bunch = data[method]["bunch"][-1]

        ymax_global = 0.0
        for i in range(4):
            values, edges = np.histogram(bunch[:, i], bins=bins, range=limits[i])
            ymax_global = max(ymax_global, np.max(values))

        vmax_global = np.zeros((4, 4))
        for i in range(4):
            for j in range(i):
                axis = (j, i)
                values, edges = np.histogramdd(
                    bunch[:, (j, i)], bins=bins, range=(limits[j], limits[i])
                )
                values = scipy.ndimage.gaussian_filter(values, blur)
                vmax_global[i, j] = max(vmax_global[i, j], np.max(values))

        # Plot
        for index in range(len(turns_list)):
            centroid = data[method]["centroid"][index]
            bunch = data[method]["bunch"][index]
            turn = turns_list[index]

            grid = CornerGrid(ndim=4, limits=limits, labels=labels)

            for i in range(4):
                for j in range(i + 1):
                    ax = grid.axs[i, j]
                    if i == j:
                        ax.hist(
                            bunch[:, i],
                            bins=bins,
                            range=limits[i],
                            histtype="step",
                            color="black",
                            lw=1.3,
                        )
                    else:
                        values, edges = np.histogramdd(
                            bunch[:, (j, i)], bins=bins, range=(limits[j], limits[i])
                        )
                        values = scipy.ndimage.gaussian_filter(values, blur)
                        ax.pcolormesh(
                            edges[0], edges[1], values.T, cmap="Greys", vmax=None
                        )

            for i in range(4):
                grid.axs[i, i].set_ylim(0.0, ymax_global * 1.2)

            for i in range(4):
                for j in range(i):
                    ax = grid.axs[i, j]
                    ax.scatter(
                        centroid[j],
                        centroid[i],
                        c="red",
                        s=8.0,
                    )

            grid.axs[1, 2].annotate(
                "t = {:0.2f}".format(float(turn / args.nturns)),
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                horizontalalignment="center",
                verticalalignment="center",
                color="black",
            )

            # Save figure
            plt.savefig(os.path.join(output_subdir, f"fig_{turn:05.0f}.png"), dpi=200)
            plt.close()
