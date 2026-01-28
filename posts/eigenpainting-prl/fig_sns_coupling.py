import argparse
import os

import numpy as np
import matplotlib.animation
import matplotlib.colors
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.stats
from tqdm import tqdm

from coupling import calc_eigtunes
from coupling import calc_eigvecs
from plot import CornerGridNoDiag
from plot import plot_vector
from utils import track

plt.style.use("style.mplstyle")

        

def animate_corner(
    particles: np.ndarray, 
    vectors: list[list[np.ndarray]] = None, 
    limits: list[tuple[float, float]] = None, 
    vector_kws: dict = None, 
    **kws
) -> matplotlib.animation.FuncAnimation:
    
    # Set plot limits.
    if limits is None:
        xmax = np.max(particles, axis=0)
        xmax = xmax * 1.4
        xmax[0] = xmax[2] = max(xmax[0], xmax[2])
        xmax[1] = xmax[3] = max(xmax[1], xmax[3])
        limits = list(zip(-xmax, xmax))

    # Set plot labels.
    labels = ["x", "x'", "y", "y'"]

    # Set default key word arguments.
    if vector_kws is None:
        vector_kws = dict()
    vector_kws.setdefault("head_width", 0.2)
    vector_kws.setdefault("head_length", 0.4)
    
    kws.setdefault("marker", ".")
    kws.setdefault("mec", "None")
    kws.setdefault("lw", 0.0)
    kws.setdefault("color", "black")
    kws.setdefault("ms", 5.0)

    # Create figure
    grid = CornerGridNoDiag(ndim=4, figwidth=6.0, limits=limits, labels=labels)

    new_lines = [[], [], []]
    old_lines = [[], [], []]
    for i in range(3):
        for j in range(i + 1):
            ax = grid.axs[i, j]
            
            old_line, = ax.plot([], [], alpha=0.25, **kws)
            old_lines[i].append(old_line)
            
            new_line, = ax.plot([], [], **kws)
            new_lines[i].append(new_line)

    plt.close()

    # Define update rule
    def update(frame):
        for ax in grid.axs.flat:
            for annotation in ax.texts:
                annotation.set_visible(False)

        x = particles[frame]
        for i in range(3):
            for j in range(i + 1):
                ax = grid.axs[i, j]

                axis = (j, i + 1)
                old_lines[i][j].set_data(particles[:frame, axis[0]], particles[:frame, axis[1]])
                new_lines[i][j].set_data((x[axis[0]],), (x[axis[1]],))
                
                if vectors is not None:
                    v1 = vectors[0][frame]
                    v2 = vectors[1][frame]
                    v1_proj = v1[[axis[0], axis[1]]]
                    v2_proj = v2[[axis[0], axis[1]]]
                    plot_vector(
                        v1_proj,
                        origin=(0, 0), 
                        color="blue", 
                        ax=ax, 
                        **vector_kws
                    )
                    plot_vector(
                        v2_proj,
                        origin=v1_proj,
                        color="red",
                        ax=ax,
                        **vector_kws
                    )
                    
        grid.axs[0, 1].annotate(
            "Period {}".format(frame),
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            horizontalalignment="center",
        )

    return matplotlib.animation.FuncAnimation(grid.fig, update, frames=particles.shape[0])


# Setup
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Load transfer matrix.
M = np.loadtxt("data/matrix.txt")

# Calculate eigenvectors.
v1, v2 = calc_eigvecs(M)
nu1, nu2 = calc_eigtunes(M)


# Plot particle orbits
# --------------------------------------------------------------------------------------

# Set initial particle coordinates.
J1 = 0.5 * 25.0  # mode 1 amplitude
J2 = 0.5 * 25.0  # mode 2 amplitude
psi1 = np.pi * 0.00  # mode 1 phase
psi2 = np.pi * 0.25  # mode 2 phase
x1 = np.real(np.sqrt(2.0 * J1) * v1 * np.exp(1.0j * psi2))
x2 = np.real(np.sqrt(2.0 * J2) * v2 * np.exp(1.0j * psi2))
x = x1 + x2

# Track particles 1000 turns and store turn-by-turn coordinates.
turns = 1000
coords1 = track(M, x1, turns)
coords2 = track(M, x2, turns)
coords = track(M, x, turns)

# Calculate plot limits.
xmax = 1.4 * np.max(coords, axis=0)
xmax[0] = xmax[2] = max(xmax[0], xmax[2])
xmax[1] = xmax[3] = max(xmax[1], xmax[3])
limits = list(zip(-xmax, xmax))

labels = ["x [mm]", "x' [mrad]", "y [mm]", "y' [mrad]"]

# Plot corner
grid = CornerGridNoDiag(ndim=4, figwidth=6.0, limits=limits, labels=labels)
grid.plot_scatter(coords, marker=".", s=5, ec="none", color="black")
plt.savefig(os.path.join(output_dir, "fig_corner.png"), dpi=250)
plt.close()

turns_plot = 45
animation = animate_corner(coords[:turns_plot], limits=limits);
animation.save(os.path.join(output_dir, "fig_corner.gif"), dpi=250)

# Plot corner with eigenvectors
grid = CornerGrid(ndim=4, figwidth=6.0, limits=limits, labels=labels)
grid.plot_scatter(coords, marker=".", s=5, ec="none", color="lightgrey")
grid.plot_scatter(coords1, marker=".", s=5, ec="none", color="red")
grid.plot_scatter(coords2, marker=".", s=5, ec="none", color="blue")
plt.savefig(os.path.join(output_dir, "fig_corner_eig_ellipse.png"), dpi=250)
plt.close()

animation = animate_corner(coords[:turns_plot], limits=limits, vectors=[coords1, coords2])
animation.save(os.path.join(output_dir, "fig_corner_vec.gif"), dpi=250)
