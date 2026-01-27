import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["figure.constrained_layout.use"] = True


def poisson_matrix(ndim: int) -> np.ndarray:
    U = np.zeros((ndim, ndim))
    for i in range(0, ndim, 2):
        U[i : i + 2, i : i + 2] = [[0.0, 1.0], [-1.0, 0.0]]
    return U


def intrinsic_emittances(cov_matrix: np.ndarray) -> tuple[float, ...]:
    S = cov_matrix.copy()
    U = poisson_matrix(ndim)
    tr_SU2 = np.trace(np.linalg.matrix_power(np.matmul(S, U), 2))
    det_S = np.linalg.det(S)
    eps_1 = 0.5 * np.sqrt(-tr_SU2 + np.sqrt(tr_SU2**2 - 16.0 * det_S))
    eps_2 = 0.5 * np.sqrt(-tr_SU2 - np.sqrt(tr_SU2**2 - 16.0 * det_S))
    return (eps_1, eps_2)


def normalize_eigvec(v: np.ndarray) -> np.ndarray:
    v = np.copy(v)
    U = poisson_matrix(len(v))
    norm = np.abs(np.linalg.multi_dot([np.conj(v), U, v]).imag)
    return v * np.sqrt(2.0 / np.abs(norm))


def norm_matrix_from_eigvecs(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    V = np.zeros((4, 4))
    V[:, 0] = +np.real(v1)
    V[:, 1] = -np.imag(v1)
    V[:, 2] = +np.real(v2)
    V[:, 3] = -np.imag(v2)
    return np.linalg.inv(V)


def norm_matrix_from_cov(cov_matrix: np.ndarray) -> np.ndarray:
    S = cov_matrix
    U = poisson_matrix(cov_matrix.shape[0])
    eig_res = np.linalg.eig(np.matmul(S, U))
    v1 = eig_res.eigenvectors[:, 0]
    v2 = eig_res.eigenvectors[:, 2]
    return norm_matrix_from_eigvecs(v1, v2)


def rms_ellipse_params(
    cov_matrix: np.ndarray, axis: tuple[int, int]
) -> tuple[float, float, float]:
    i, j = axis

    sii = cov_matrix[i, i]
    sjj = cov_matrix[j, j]
    sij = cov_matrix[i, j]

    phi = -0.5 * np.arctan2(2 * sij, sii - sjj)

    _sin = np.sin(phi)
    _cos = np.cos(phi)
    _sin2 = _sin**2
    _cos2 = _cos**2

    c1 = np.sqrt(abs(sii * _cos2 + sjj * _sin2 - 2 * sij * _sin * _cos))
    c2 = np.sqrt(abs(sii * _sin2 + sjj * _cos2 + 2 * sij * _sin * _cos))

    return (c1, c2, phi)


def rotation_matrix_xy(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array(
        [
            [c, 0, s, 0],
            [0, c, 0, s],
            [-s, 0, c, 0],
            [0, -s, 0, c],
        ]
    )
    return R


def get_limits(points: np.ndarray) -> list[tuple[float, float]]:
    xmax = 4.0 * np.std(points, axis=0)
    xmax[0] = xmax[2] = max(xmax[0], xmax[2])
    xmax[1] = xmax[3] = max(xmax[1], xmax[3])
    return list(zip(-xmax, xmax))


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

    def set_labels(self, labels: list[str] = None, **kws) -> None:
        if labels is None:
            return

        self.labels = labels
        for i in range(self.ndim):
            for j in range(self.ndim):
                ax = self.axs[i, j]
                if j == 0:
                    ax.set_ylabel(labels[i], **kws)
                if i == self.ndim - 1:
                    ax.set_xlabel(labels[j], **kws)

        self.axs[0, 0].set_ylabel("")

        self.fig.align_xlabels()
        self.fig.align_ylabels()

    def plot_scatter(self, points: np.ndarray, diag_kws: dict = None, **kws):
        if diag_kws is None:
            diag_kws = dict()

        diag_kws.setdefault("color", "black")
        diag_kws.setdefault("histtype", "stepfilled")
        diag_kws.setdefault("bins", 45)

        for i in range(self.ndim):
            for j in range(i + 1):
                ax = self.axs[i, j]
                if j < i:
                    ax.scatter(points[:, j], points[:, i], **kws)
                else:
                    ax.hist(points[:, j], range=ax.get_xlim(), **diag_kws)

    def plot_cov(self, cov_matrix: np.ndarray, **kws):
        kws.setdefault("color", "black")
        kws.setdefault("fill", False)
        for i in range(self.ndim):
            for j in range(i):
                ax = self.axs[i, j]

                cx, cy, angle = rms_ellipse_params(cov_matrix, axis=(j, i))
                angle = -np.degrees(angle)
                patch = patches.Ellipse(
                    xy=(0.0, 0.0),
                    width=(4.0 * cx),
                    height=(4.0 * cy),
                    angle=angle,
                    **kws
                )
                ax.add_patch(patch)


rng = np.random.default_rng(0)

eps_x = 0.25
eps_y = 0.25

beta_x = 5.0
beta_y = 2.0
alpha_x = +2.0
alpha_y = -0.2
gamma_x = (1.0 + alpha_x**2) / beta_x
gamma_y = (1.0 + alpha_y**2) / beta_y

cov_matrix = np.eye(4)
cov_matrix[0, 0] = beta_x * eps_x
cov_matrix[2, 2] = beta_y * eps_y
cov_matrix[0, 1] = -alpha_x * eps_x
cov_matrix[2, 3] = -alpha_y * eps_y
cov_matrix[1, 1] = gamma_x * eps_x
cov_matrix[3, 3] = gamma_y * eps_y

V = np.identity(4)
V[0, 0] = beta_x
V[1, 0] = -alpha_x
V[2, 2] = beta_y
V[3, 2] = -alpha_y
V[0:2, 0:2] /= np.sqrt(beta_x)
V[2:4, 2:4] /= np.sqrt(beta_y)

X = rng.normal(size=(1000, 4))
X = np.matmul(X, V.T)

M = rotation_matrix_xy(angle=np.radians(20.0))
X = np.matmul(X, M.T)
S = np.cov(X.T)

xrms = np.std(X, axis=0)


labels = [r"$x$", r"$x'$", r"$y$", r"$y'$"]
limits = get_limits(X)

grid = CornerGrid(ndim=4, figwidth=5.5)
grid.set_labels(labels)
grid.set_limits(limits)
grid.plot_scatter(X, s=1, c="grey", ec="none", diag_kws=dict(color="grey"))
grid.plot_cov(S, color="red")
grid.axs[1, 0].annotate(
    r"$\varepsilon_x$",
    xy=(3.0, 0.5),
    color="red",
)
grid.axs[3, 2].annotate(
    r"$\varepsilon_y$",
    xy=(4.5, 0.0),
    color="red",
)
plt.savefig("images/cov.png", dpi=300)

V_inv = norm_matrix_from_cov(S)
X = np.matmul(X, V_inv.T)
S = np.cov(X.T)

xrms = np.std(X, axis=0)

labels=[
    r"$u_1$",
    r"$u_1'$",
    r"$u_2$",
    r"$u_2'$",
]
limits = get_limits(X)

grid = CornerGrid(ndim=4, figwidth=5.5)
grid.set_labels(labels)
grid.set_limits(limits)
grid.plot_scatter(X, s=1, c="grey", ec="none", diag_kws=dict(color="grey"))
grid.plot_cov(S, color="red")
grid.axs[1, 0].annotate(
    r"$\varepsilon_1$",
    xy=(2.0 * xrms[0] + 1.5, 1.5),
    color="red",
)
grid.axs[3, 2].annotate(
    r"$\varepsilon_2$",
    xy=(2.0 * xrms[2] + 1.5, 0.0),
    color="red",
)
plt.savefig("images/cov_norm.png", dpi=300)
