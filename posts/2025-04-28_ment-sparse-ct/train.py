"""Test 2D MENT with high-resolution image."""
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import ment  # https://github.com/austin-hoover/ment
from tqdm import tqdm

from utils import load_image
from utils import rec_fbp
from utils import rec_sart
from utils import radon_transform


plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["image.cmap"] = "Blues"
plt.rcParams["savefig.dpi"] = 700.0
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True


# Arguments
# --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--nmeas", type=int, default=25)
parser.add_argument("--iters", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.33)
parser.add_argument("--sart-iters", type=int, default=5)
args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

ndim = 2
nmeas = args.nmeas

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


# Ground truth image
# --------------------------------------------------------------------------------------

res = 256
image_true = load_image(res=res)

xmax = 1.0
grid_edges = 2 * [np.linspace(-xmax, xmax, res + 1)]
grid_coords = [0.5 * (e[:-1] + e[1:]) for e in grid_edges]
grid_points = np.vstack([c.ravel() for c in np.meshgrid(*grid_coords, indexing="ij")]).T


# Forward model
# --------------------------------------------------------------------------------------

angles = np.linspace(0.0, np.pi, args.nmeas, endpoint=False)

transforms = []
for angle in angles:
    matrix = ment.utils.rotation_matrix(angle)
    transform = ment.sim.LinearTransform(matrix)
    transforms.append(transform)


# Training data
# --------------------------------------------------------------------------------------

sinogram = radon_transform(image_true, angles)

projections = []
for j in range(sinogram.shape[1]):
    projection = ment.Histogram1D(
        values=sinogram[:, j],
        coords=grid_coords[0],
        axis=0,
        thresh=0.001,
        thresh_type="frac",
    )
    projection.normalize()
    projections.append([projection])


# Reconstruction model
# --------------------------------------------------------------------------------------

prior = ment.GaussianPrior(ndim=2, scale=10.0)

integration_limits = [[(-xmax, xmax)] for _ in range(nmeas)]
integration_size = image_true.shape[1]

model = ment.MENT(
    ndim=ndim,
    transforms=transforms,
    projections=projections,
    prior=prior,
    sampler=None,
    integration_limits=integration_limits,
    integration_size=integration_size,
    integration_loop=False,
    mode="integrate",
    verbose=0,
)


# Training
# --------------------------------------------------------------------------------------

for iteration in tqdm(range(args.iters)):
    model.gauss_seidel_step(learning_rate=args.lr)


# Plot results
# --------------------------------------------------------------------------------------

# Make dictionary for comparison:
results = {}
for method in ["fbp", "sart", "ment", "true"]:
    results[method] = {}
    for key in ["sinogram", "image"]:
        results[method][key] = None

# TRUE
image_true = image_true.copy()
sinogram_true = radon_transform(image_true, angles)
results["true"]["sinogram"] = sinogram_true.copy()
results["true"]["image"] = image_true.copy()

# MENT
image_pred = model.prob(grid_points).reshape(image_true.shape)
sinogram_pred = radon_transform(image_pred, angles=angles)
results["ment"]["image"] = image_pred.copy()
results["ment"]["sinogram"] = sinogram_pred.copy()

# FBP
image_pred = rec_fbp(sinogram_true, angles)
sinogram_pred = radon_transform(image_pred, angles=angles)
results["fbp"]["image"] = image_pred.copy()
results["fbp"]["sinogram"] = sinogram_pred.copy()

# SART
image_pred = rec_sart(sinogram_true, angles, iterations=args.sart_iters)
sinogram_pred = radon_transform(image_pred, angles=angles)
results["sart"]["image"] = image_pred.copy()
results["sart"]["sinogram"] = sinogram_pred.copy()

# Normalize and scale
for key in ["image", "sinogram"]:
    for name in results:
        results[name][key] /= np.sum(results[name][key])
    for name in results:
        results[name][key] /= np.max(results["true"][key])

# Plot images and sinograms
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(9.0, 4.5), gridspec_kw=dict(wspace=0, hspace=0), tight_layout=True)
for j, name in enumerate(results):
    for i, key in enumerate(["image", "sinogram"]):
        axs[0, j].pcolormesh(results[name]["image"].T,  vmin=0.0, vmax=1.0)
        axs[1, j].pcolormesh(results[name]["sinogram"], vmin=0.0, vmax=1.0)
    axs[0, j].set_title(name.upper()) 
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig(os.path.join(output_dir, "fig_compare_all.png"))