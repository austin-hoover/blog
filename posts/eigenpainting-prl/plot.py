import matplotlib.animation
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage


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


class CornerGridNoDiag:
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


class CornerGrid:
    def __init__(
        self,
        ndim: int,
        limits: list[tuple[float, float]] = None,
        labels: list[str] = None,
        figwidth: float = None,
    ) -> None:
        self.ndim = ndim

        if figwidth is None:
            figwidth = 1.7 * ndim

        self.fig, self.axs = plt.subplots(
            ncols=ndim,
            nrows=ndim,
            figsize=(figwidth, figwidth),
            sharex=False,
            sharey=False,
        )

        for i in range(self.ndim - 1):
            for ax in self.axs[i, :]:
                ax.set_xticklabels([])

        for j in range(1, self.ndim):
            for ax in self.axs[:, j]:
                ax.set_yticklabels([])

        for i in range(self.ndim):
            self.axs[i, i].set_yticks([])

        for ax in self.axs.flat:
            for loc in ["top", "right"]:
                ax.spines[loc].set_visible(False)

        for i in range(self.ndim):
            for j in range(self.ndim):
                if i < j:
                    self.axs[i, j].axis("off")

        self.limits = None
        self.labels = None
        self.set_limits(limits)
        self.set_labels(labels)

    def set_limits(self, limits: list[tuple[float, float]] = None) -> None:
        if limits is None:
            return

        self.limits = limits
        for i in range(4):
            for j in range(4):
                ax = self.axs[i, j]
                ax.set_xlim(limits[j])
                if i != j:
                    ax.set_ylim(limits[i])

    def set_labels(self, labels: list[str] = None) -> None:
        if labels is None:
            return

        self.labels = labels
        for i in range(self.ndim):
            for j in range(self.ndim):
                ax = self.axs[i, j]
                if j == 0:
                    ax.set_ylabel(labels[i])
                if i == self.ndim - 1:
                    ax.set_xlabel(labels[j])

        self.fig.align_xlabels()
        self.fig.align_ylabels()