import os
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.stats
from tqdm import tqdm

plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["figure.constrained_layout.use"] = True


def rotation_matrix(angle: float) -> np.ndarray:
    matrix = np.zeros((2, 2))
    matrix[0, 0] = +np.cos(angle)
    matrix[1, 1] = +np.cos(angle)
    matrix[0, 1] = +np.sin(angle)
    matrix[1, 0] = -np.sin(angle)
    return matrix


def truncate_cmap(cmap, left=0.0, right=1.0, n=100):
    string = "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=left, b=right)
    values = cmap(np.linspace(left, right, n))
    return matplotlib.colors.LinearSegmentedColormap.from_list(string, values)


def cubehelix_cmap(color="red", dark=0.20):
    kws = dict(
        n_colors=12,
        rot=0.0,
        gamma=1.0,
        hue=1.0,
        light=1.0,
        dark=dark,
        as_cmap=True,
    )

    cmap = None
    if color == "red":
        cmap = sns.cubehelix_palette(start=0.9, **kws)
    elif color == "pink":
        cmap = sns.cubehelix_palette(start=0.8, **kws)
    elif color == "blue":
        cmap = sns.cubehelix_palette(start=2.8, **kws)
    else:
        raise ValueError
    return cmap


class Painter:
    """Runs painting simulation space painting.

    (Linear approximation, non-interacting particles, normalized phase space.)
    """

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

    def generate_minipulse(self) -> np.ndarray:
        x = scipy.stats.truncnorm.rvs(
            scale=self.inj_rms,
            size=(self.n_inj, 4),
            a=-self.inj_cut,
            b=+self.inj_cut,
        )
        return x

    def paint(self, nturns: list[int]) -> np.ndarray:
        # Generate `n_turns` minipulses at the origin.
        minipulses = [self.generate_minipulse() for _ in range(nturns + 1)]

        # Move each minipulse to its final amplitude.
        for t in range(nturns + 1):
            minipulses[t] += self.get_inj_coords(t)

        # Rotate each minipulse by the requested number of turns.
        for t, minipulse in enumerate(tqdm(minipulses)):
            matrix = np.zeros((4, 4))
            matrix[0:2, 0:2] = rotation_matrix(2.0 * np.pi * self.tune_x * t)
            matrix[2:4, 2:4] = rotation_matrix(2.0 * np.pi * self.tune_y * t)
            minipulses[t] = np.matmul(minipulses[t], matrix.T)

        bunch = np.vstack(minipulses)
        return bunch


    
# Settings
# --------------------------------------------------------------------------------

tune_x = 0.1810201  # horizontal tune
tune_y = tune_x - 0.143561  # vertical tune
n_turns = 2800  # number of turns to paint
n_inj = 200  # number of particles per turn

# Plot only these turns
stride = 200
turns_list = list(range(0, n_turns + stride, stride))

# Create simulator
# --------------------------------------------------------------------------------

painter = Painter(
    tune_x=tune_x,
    tune_y=tune_y,
    n_turns=n_turns,
    n_inj=n_inj,
    inj_rms=0.10,
)

# Run simulations
# --------------------------------------------------------------------------------

bunches = {}
centroids = {}

# Anti-correlated painting
painter.method = "anti-correlated"
painter.inj_xmax = np.array([1.0, 0.0, 1.0, 0.0])
bunches["anti-correlated"] = [painter.paint(t) for t in turns_list]
centroids["anti-correlated"] = [painter.get_inj_coords(t) for t in turns_list]

# Correlated painting
painter.method = "correlated"
painter.inj_xmax = np.array([1.0, 0.0, 1.0, 0.0])
bunches["correlated"] = [painter.paint(t) for t in turns_list]
centroids["correlated"] = [painter.get_inj_coords(t) for t in turns_list]

# Eigenpainting (correlated painting along eigenvector)
painter.method = "correlated"
painter.inj_xmax = np.array([1.0, 0.0, 0.0, 1.0])  # eigenvector
painter.tune_x = tune_x
painter.tune_y = tune_x
bunches["eig"] = [painter.paint(t) for t in turns_list]
centroids["eig"] = [painter.get_inj_coords(t) for t in turns_list]


# Create GIF
# --------------------------------------------------------------------------------

os.makedirs("outputs", exist_ok=True)

key = "correlated"

bins = 64
limits = [(-2.0, 2.0), (-2.0, 2.0)]

cmap = "gray_r"
colors = [(i, i, i, 1.0) for i in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]]
rng = np.random.default_rng(123)


# for index in range(len(turns_list)):
    
    
#     turn = turns_list[index]
    
#     bunch = bunches[key][index]
    
#     fig, ax = plt.subplots(figsize=(4, 4))
#     hist, edges = np.histogramdd(bunch[:, (0, 2)], bins=bins, range=limits)
#     hist = scipy.ndimage.gaussian_filter(hist, 1.0)
#     hist = hist / np.max(hist)
#     coords = [0.5 * (e[:-1] + e[1:]) for e in edges]
#     # ax.contourf(coords[0], coords[1], hist.T, levels=5, colors=colors)
#     ax.pcolormesh(coords[0], coords[1], hist.T, cmap=cmap)
    
#     # for centroid in centroids[key][:index]:
#     centroid = centroids[key][index]
#     ax.scatter(
#         centroid[0],
#         centroid[2],
#         color="red",
#         s=10.0,
#     )
    
#     ax.quiver(
#         centroid[0],
#         centroid[2],
#         centroid[1],
#         centroid[3],
#         color="red",
#         scale=10.0,
#     )
#     ax.set_xlim(limits[0])
#     ax.set_ylim(limits[0])
    
#     plt.savefig(f"outputs/fig_{turn:05.0f}.png", dpi=200)
#     plt.close()
    

limits = 4 * [(-2.0, 2.0)]
for index in range(len(turns_list)):
    bunch = bunches[key][index]
    turn = turns_list[index]
    
    fig, axs = plt.subplots(figsize=(8, 8), sharex=False, sharey=False, nrows=4, ncols=4)

    for i in range(4):
        for j in range(4):
            ax = axs[i, j]
            ax.set_xlim(limits[j])
            if i != j:
                ax.set_ylim(limits[i])

    for i in range(3):
        for ax in axs[i, :]:
            ax.set_xticks([])
    for j in range(1, 4):
        for ax in axs[:, j]:
            ax.set_yticks([])
                    
    for i in range(4):
        for j in range(4):
            ax = axs[i, j]

            axis = (j, i)

            if i == j:
                ax.hist(bunch[:, i], bins=bins, range=limits[i], histtype="step", color="black")

            elif i > j:
                hist, edges = np.histogramdd(bunch[:, axis], bins=bins, range=[limits[k] for k in axis])
                hist = scipy.ndimage.gaussian_filter(hist, 1.0)
                hist = hist / np.max(hist)
                coords = [0.5 * (e[:-1] + e[1:]) for e in edges]
                # ax.contourf(coords[0], coords[1], hist.T, levels=5, colors=colors)
                ax.pcolormesh(coords[0], coords[1], hist.T, cmap=cmap)
            
                centroid = centroids[key][index]
                ax.scatter(
                    centroid[j],
                    centroid[i],
                    color="red",
                    s=10.0,
                )
            else:
                ax.axis("off")

    axs[0, 0].set_yticks([])

    for ax in axs.flat:
        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
    
    plt.savefig(f"outputs/fig_{turn:05.0f}.png", dpi=200)
    plt.close()
    
    
    








    

    

    