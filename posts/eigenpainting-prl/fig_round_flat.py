import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["figure.constrained_layout.use"] = True


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
        

grid = CornerGrid(ndim=4, limits=(4 * [(-4.0, 4.0)]), labels=["x", "x'", "y", "y'"])







