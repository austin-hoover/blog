import argparse
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from tqdm import tqdm

from coupling import calc_eigvecs
from coupling import calc_eigtunes
from coupling import build_norm_matrix_from_tmat
from plot import CornerGrid
from utils import NormalizedPainter

plt.style.use("style.mplstyle")


# Parse arguments
# --------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--turns", type=int, default=2800)
parser.add_argument("--stride", type=int, default=200)
parser.add_argument("--inj-size", type=int, default=200)
parser.add_argument("--inj-rms", type=float, default=0.10)
parser.add_argument("--blur", type=float, default=1.0)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)

# Load transfer matrix
M = np.loadtxt("data/matrix.txt")
V = np.linalg.inv(build_norm_matrix_from_tmat(M))
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

turns_list = list(range(0, args.turns + args.stride, args.stride))

data = {}
for method in ["corr", "anticorr", "flat", "eigen"]:
    data[method] = {}
    for key in ["bunch", "bunch_n", "inj_point", "inj_point_n"]:
        data[method][key] = []


# Run simulations
# --------------------------------------------------------------------------------


def run_sim(painter: NormalizedPainter, turns_list: list[int]) -> dict:
    data = {}
    for key in ["bunch", "bunch_n", "inj_point", "inj_point_n"]:
        data[key] = []

    for t in turns_list:
        bunch_n = painter.paint(t)
        bunch = np.matmul(bunch_n, V.T)

        inj_point_n = painter.get_inj_point(t)
        inj_point = np.matmul(V, inj_point_n)

        data["bunch"].append(bunch)
        data["bunch_n"].append(bunch_n)
        data["inj_point"].append(inj_point)
        data["inj_point_n"].append(inj_point_n)

    return data


# Set maximum amplitudes and phases.
J1 = 25.0  # mode 1 amplitude
J2 = 25.0  # mode 1 amplitude
psi1 = np.pi * 0.0  # mode 1 phase
psi2 = np.pi * 0.5  # mode 2 phase

# Calculate maximum phase space coordinates.
umax = np.zeros(4)
umax[0] = np.sqrt(2.0 * J1) * np.cos(psi1)
umax[1] = np.sqrt(2.0 * J1) * np.sin(psi1)
umax[2] = np.sqrt(2.0 * J2) * np.cos(psi2)
umax[3] = np.sqrt(2.0 * J2) * np.sin(psi2)
painter.set_umax(umax)


# Run correlated painting simulation.
painter.method = "corr"
data["corr"] = run_sim(painter, turns_list)

# Run anti-correlated painting simulation.
painter.method = "anticorr"
data["anticorr"] = run_sim(painter, turns_list)

# Run eigenpainting simulation (correlated painting with J2=0).
J2 = 0.0
umax[0] = np.sqrt(2.0 * J1) * np.cos(psi1)
umax[1] = np.sqrt(2.0 * J1) * np.sin(psi1)
umax[2] = np.sqrt(2.0 * J2) * np.cos(psi2)
umax[3] = np.sqrt(2.0 * J2) * np.sin(psi2)
painter.set_umax(umax)
painter.method = "corr"

data["eig"] = run_sim(painter, turns_list)


# Plot data
# ----------------------------------------------------------------------


def plot_bunch(
    bunch: np.ndarray,
    inj_point: np.ndarray,
    t: float,
    limits: list[tuple[float, float]],
    yscale: float = None,
    blur: float = 0.0,
) -> CornerGrid:

    grid = CornerGrid(ndim=4, limits=limits)
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
                ax.pcolormesh(edges[0], edges[1], values.T, cmap="Greys", vmax=None)

    if ymax:
        for i in range(4):
            grid.axs[i, i].set_ylim(0.0, ymax * 1.2)

    for i in range(4):
        for j in range(i):
            ax = grid.axs[i, j]
            ax.scatter(
                inj_point[j],
                inj_point[i],
                c="red",
                s=8.0,
            )

    grid.axs[1, 2].annotate(
        "t = {:0.2f}".format(t),
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        horizontalalignment="center",
        verticalalignment="center",
        color="black",
    )
    return grid


# Settings
bins = 64
cmap = "gray_r"
blur = args.blur

# Plot phase space distribution
last_bunch = data["corr"]["bunch"][-1]
xmax = 3.5 * np.std(last_bunch, axis=0)
xmax[0] = xmax[2] = max(xmax[0], xmax[2])
xmax[1] = xmax[3] = max(xmax[1], xmax[3])
limits = list(zip(-xmax, xmax))

labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]

for method in data:
    if not data[method]["bunch"]:
        continue

    output_subdir = os.path.join(output_dir, method)
    os.makedirs(output_subdir, exist_ok=True)

    last_bunch = data[method]["bunch"][-1]
    ymax = 0.0
    for i in range(4):
        values, edges = np.histogram(last_bunch[:, i], bins=bins, range=limits[i])
        ymax = max(ymax, np.max(values))

    for index in range(len(turns_list)):
        bunch = data[method]["bunch"][index]
        turn = turns_list[index]
        inj_point = data[method]["inj_point"][index]

        grid = plot_bunch(
            bunch=bunch,
            inj_point=inj_point,
            limits=limits,
            t=float(turn / args.turns),
            blur=blur,
        )
        grid.set_labels(labels)

        plt.savefig(os.path.join(output_subdir, f"fig_{turn:05.0f}.png"), dpi=200)
        plt.close()


# Plot phase space distribution (normalized coordinates)

last_bunch = data["corr"]["bunch_n"][-1]
xmax = 3.5 * np.std(last_bunch, axis=0)
xmax[0] = xmax[2] = max(xmax[0], xmax[2])
xmax[1] = xmax[3] = max(xmax[1], xmax[3])
limits = list(zip(-xmax, xmax))

labels = [r"$u_1$", r"$u_1'$", r"$u_2$", r"$u_2'$"]

for method in data:
    if not data[method]["bunch_n"]:
        continue

    output_subdir = os.path.join(output_dir, method + "_n")
    os.makedirs(output_subdir, exist_ok=True)

    last_bunch = data[method]["bunch_n"][-1]
    ymax = 0.0
    for i in range(4):
        values, edges = np.histogram(last_bunch[:, i], bins=bins, range=limits[i])
        ymax = max(ymax, np.max(values))

    for index in range(len(turns_list)):
        bunch = data[method]["bunch_n"][index]
        turn = turns_list[index]
        inj_point = data[method]["inj_point_n"][index]

        grid = plot_bunch(
            bunch=bunch,
            inj_point=inj_point,
            limits=limits,
            t=float(turn / args.turns),
            blur=blur,
        )
        grid.set_labels(labels)

        plt.savefig(os.path.join(output_subdir, f"fig_{turn:05.0f}.png"), dpi=200)
        plt.close()
