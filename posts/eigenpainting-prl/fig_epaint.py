import argparse
import os

import numpy as np
import matplotlib.animation
import matplotlib.colors
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.stats
from tqdm import tqdm

plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["figure.constrained_layout.use"] = True


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
    eig_res = np.linalg.eig(self.M)
    v1 = normalize_eigvec(eig_res.eigenvectors[:, 0])
    v2 = normalize_eigvec(eig_res.eigenvectors[:, 2])
    return build_norm_matrix_from_eigvecs(v1, v2)


def track(matrix: np.ndarray, x: np.ndarray, turns: int) -> np.ndarray:
    coords = np.zeros((turns + 1, 4))
    coords[0, :] = np.copy(x)
    for i in range(turns):
        x = np.matmul(matrix, x)
        coords[i + 1, :] = np.copy(x)
    return coords


class CornerGrid:
    def __init__(self, ndim: int, limits: list[tuple[float, float]] = None, labels: list[str] = None, figwidth: float = None) -> None:
        self.ndim = ndim       

        # Create figure
        if figwidth is None:
            figwidth = 2.0 * (self.ndim - 1)
            
        self.fig, self.axs = plt.subplots(ncols=(ndim - 1), nrows=(ndim - 1), figsize=(figwidth, figwidth), sharex="col", sharey="row")

        # Turn off upper-diagonal plots.
        for i in range(self.ndim - 1):
            for j in range(self.ndim - 1):
                if i < j:
                    self.axs[i, j].axis("off")

        # Remove top/right spines.
        for ax in self.axs.flat:
            for loc in ["top", "right"]:
                ax.spines[loc].set_visible(False)

        # Set axis limits and labels.
        self.limits = None
        self.labels = None
        self.set_limits(limits)
        self.set_labels(labels)
        
    def set_limits(self, limits: list[tuple[float, float]] = None) -> None:
        if limits is None:
            return
            
        self.limits = limits
        for j in range(self.ndim - 1):
            for ax in self.axs[:, j]:
                ax.set_xlim(self.limits[j])
        for i in range(self.ndim - 1):
            for ax in self.axs[i, :]:
                ax.set_ylim(self.limits[i + 1])

    def set_labels(self, labels: list[str] = None) -> None:
        if labels is None:
            return
            
        self.labels = labels
        for j in range(self.ndim - 1):
            self.axs[-1, j].set_xlabel(self.labels[j])
        for i in range(self.ndim - 1):
            self.axs[i, 0].set_ylabel(self.labels[i + 1])

        self.fig.align_xlabels()
        self.fig.align_ylabels()
            
    def plot_scatter(self, points: np.ndarray, **kws) -> None:
        for i in range(self.ndim - 1):
            for j in range(self.ndim - 1):
                ax = self.axs[i, j]
                if i >= j:
                    ax.scatter(points[:, j], points[:, i + 1], **kws)
        self.set_limits(self.limits)

        
def plot_vector(
    vector: np.ndarray,
    origin: tuple[float] = (0.0, 0.0),
    color: str = "black",
    lw: float = None,
    style: str = "->",
    head_width: float = 0.4,
    head_length: float = 0.8,
    ax=None,
) -> None:
    props = dict()
    props["arrowstyle"] = f"{style},head_width={head_width},head_length={head_length}"
    props["shrinkA"] = props["shrinkB"] = 0
    props["fc"] = props["ec"] = color
    props["lw"] = lw

    vector = np.copy(vector)
    vector = np.add(vector, origin)
    ax.annotate("", xy=(vector[0], vector[1]), xytext=origin, arrowprops=props)


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
    grid = CornerGrid(ndim=4, figwidth=6.0, limits=limits, labels=labels)

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


output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


# Load transfer matrix.
M = np.loadtxt("data/matrix.txt")

# Calculate eigenvectors.
eigres = np.linalg.eig(M)
v1 = normalize_eigvec(eigres.eigenvectors[:, 0])
v2 = normalize_eigvec(eigres.eigenvectors[:, 2])

# Set initial particle coordinates.
J1 = 0.5 * 25.0  # mode 1 amplitude
J2 = 0.5 * 25.0  # mode 2 amplitude
psi1 = np.pi * 0.00  # mode 1 phase
psi2 = np.pi * 0.25  # mode 2 phase
x_1 = np.real(np.sqrt(2.0 * J1) * v1 * np.exp(1.0j * psi2))
x_2 = np.real(np.sqrt(2.0 * J2) * v2 * np.exp(1.0j * psi2))
x = x_1 + x_2

# Track particles 1000 turns and store turn-by-turn coordinates.
turns = 1000
coords_1 = track(M, x_1, turns)
coords_2 = track(M, x_2, turns)
coords = track(M, x, turns)

# Calculate plot limits.
xmax = 1.4 * np.max(coords, axis=0)
xmax[0] = xmax[2] = max(xmax[0], xmax[2])
xmax[1] = xmax[3] = max(xmax[1], xmax[3])
limits = list(zip(-xmax, xmax))

labels = ["x [mm]", "x' [mrad]", "y [mm]", "y' [mrad]"]

grid = CornerGrid(ndim=4, figwidth=6.0, limits=limits, labels=labels)
grid.plot_scatter(coords, marker=".", s=5, ec="none", color="black")
plt.savefig(os.path.join(output_dir, "fig_corner.png"), dpi=250)
plt.close()

turns_plot = 45
animation = animate_corner(coords[:turns_plot], limits=limits);
animation.save(os.path.join(output_dir, "fig_corner.gif"), dpi=250)