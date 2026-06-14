"""
Symplectic Normalizing Flow — Maximum Likelihood Training on 2D Data
=====================================================================
A symplectic flow is a sequence of canonical (area-preserving) maps on phase
space (q, p).  Because every layer has Jacobian determinant = 1 the log-
likelihood simplifies to

    log p(x) = log p_z( f(x) )          (no log|det J| term!)

We compose two types of elementary symplectic shears:
  Type-A:  (q, p) -> (q + NN(p),  p)
  Type-B:  (q, p) -> (q,           p + NN(q))

Both are trivially invertible and have unit Jacobian determinant, so their
composition is also a valid symplectic map.

Test target: a "crescent / banana" distribution that lives on a curved
manifold in phase space — natural for Hamiltonian / symplectic geometry.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from torch.distributions import MultivariateNormal

torch.manual_seed(42)
np.random.seed(42)

# ── Colour palette ────────────────────────────────────────────────────────────
BG = "#0d0f14"
C1 = "#5ce1e6"  # cyan  – data
C2 = "#ff6b6b"  # coral – model samples
C3 = "#ffd93d"  # yellow – latent
GREY = "#2a2d36"
WHITE = "#e8eaf0"


# ── Target distribution ───────────────────────────────────────────────────────
def sample_target(n: int) -> torch.Tensor:
    """
    Crescent (banana) distribution in (q, p):
        q ~ N(0, 1)
        p | q ~ N(q²/2 – 1,  0.3²)
    This is the level-set of a harmonic oscillator potential, perturbed —
    a natural test case for symplectic flows.
    """
    q = torch.randn(n, 1)
    p_mean = 0.5 * q**2 - 1.0
    p = p_mean + 0.3 * torch.randn(n, 1)
    return torch.cat([q, p], dim=1)


# ── Symplectic building blocks ────────────────────────────────────────────────
def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class ShearA(nn.Module):
    """(q, p) -> (q + f(p), p)   — canonical shear along q"""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.f = _mlp(1, hidden, 1)

    def forward(self, q, p):
        return q + self.f(p), p

    def inverse(self, q, p):
        return q - self.f(p), p


class ShearB(nn.Module):
    """(q, p) -> (q, p + g(q))   — canonical shear along p"""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.g = _mlp(1, hidden, 1)

    def forward(self, q, p):
        return q, p + self.g(q)

    def inverse(self, q, p):
        return q, p - self.g(q)


class SymplecticFlow(nn.Module):
    """
    Stack of alternating ShearA / ShearB layers.
    det J = 1 for every layer, so log|det J| = 0 globally.
    Base distribution: standard 2-D Gaussian N(0, I).
    """

    def __init__(self, n_layers: int = 12, hidden: int = 64):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(ShearA(hidden) if i % 2 == 0 else ShearB(hidden))
        self.layers = nn.ModuleList(layers)
        self.register_buffer("mu", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))

    @property
    def base(self) -> MultivariateNormal:
        return MultivariateNormal(self.mu, self.cov)

    # data  →  latent
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        q, p = x[:, 0:1], x[:, 1:2]
        for layer in self.layers:
            q, p = layer(q, p)
        return torch.cat([q, p], dim=1)

    # latent  →  data
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        q, p = z[:, 0:1], z[:, 1:2]
        for layer in reversed(self.layers):
            q, p = layer.inverse(q, p)
        return torch.cat([q, p], dim=1)

    # log-likelihood  (no change-of-variables correction needed!)
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.base.log_prob(z)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = self.base.sample((n,))
        return self.decode(z)


# ── Training ──────────────────────────────────────────────────────────────────
def train(n_data=8000, n_layers=12, hidden=64, epochs=600, batch=512, lr=3e-3):

    data = sample_target(n_data)
    model = SymplecticFlow(n_layers=n_layers, hidden=hidden)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    losses = []
    print(f"{'Epoch':>6}  {'NLL':>10}  {'lr':>10}")
    print("─" * 32)

    for epoch in range(1, epochs + 1):
        idx = torch.randperm(n_data)[:batch]
        xb = data[idx]
        loss = -model.log_prob(xb).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        losses.append(loss.item())
        if epoch % 50 == 0 or epoch == 1:
            print(
                f"{epoch:>6}  {loss.item():>10.4f}  " f"{sched.get_last_lr()[0]:>10.2e}"
            )

    return model, data, losses


# ── Visualisation ─────────────────────────────────────────────────────────────
def visualise(model, data, losses, path):
    model.eval()
    n_vis = 4000

    with torch.no_grad():
        samples = model.sample(n_vis).numpy()
        data_latent = model.encode(data[:n_vis]).numpy()
        noise_grid = make_grid_noise()
        flow_grid = model.decode(noise_grid).numpy()

    data_np = data[:n_vis].numpy()

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.40,
        wspace=0.35,
        left=0.06,
        right=0.97,
        top=0.90,
        bottom=0.08,
    )

    ax_data = fig.add_subplot(gs[0, 0])
    ax_latent = fig.add_subplot(gs[0, 1])
    ax_samples = fig.add_subplot(gs[0, 2])
    ax_grid = fig.add_subplot(gs[1, 0])
    ax_loss = fig.add_subplot(gs[1, 1])
    ax_density = fig.add_subplot(gs[1, 2])

    style = dict(facecolor=GREY, edgecolor="none")
    for ax in [ax_data, ax_latent, ax_samples, ax_grid, ax_loss, ax_density]:
        ax.set_facecolor(GREY)
        for sp in ax.spines.values():
            sp.set_edgecolor(WHITE)
            sp.set_alpha(0.15)
        ax.tick_params(colors=WHITE, labelsize=8)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)

    # palette helpers
    def scatter(ax, xy, color, label="", alpha=0.35, s=4):
        ax.scatter(
            xy[:, 0], xy[:, 1], c=color, s=s, alpha=alpha, linewidths=0, label=label
        )

    lim = 3.5

    # 1. Target data
    scatter(ax_data, data_np, C1, "target data")
    ax_data.set_xlim(-lim, lim)
    ax_data.set_ylim(-lim, lim)
    ax_data.set_title("① Target Data  (q, p)", fontsize=11, pad=8)
    ax_data.set_xlabel("q")
    ax_data.set_ylabel("p")

    # 2. Latent space  (data encoded through flow)
    scatter(ax_latent, data_latent, C3, "encoded")
    # overlay base distribution circle
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in [1, 2, 3]:
        ax_latent.plot(
            r * np.cos(theta), r * np.sin(theta), color=WHITE, lw=0.5, alpha=0.3
        )
    ax_latent.set_xlim(-lim, lim)
    ax_latent.set_ylim(-lim, lim)
    ax_latent.set_title("② Latent Space  z = f(x)", fontsize=11, pad=8)
    ax_latent.set_xlabel("z₁")
    ax_latent.set_ylabel("z₂")

    # 3. Flow samples
    scatter(ax_samples, data_np, C1, "target", alpha=0.25)
    scatter(ax_samples, samples, C2, "flow samples", alpha=0.45)
    ax_samples.set_xlim(-lim, lim)
    ax_samples.set_ylim(-lim, lim)
    ax_samples.set_title("③ Flow Samples  x = f⁻¹(z)", fontsize=11, pad=8)
    ax_samples.set_xlabel("q")
    ax_samples.set_ylabel("p")
    ax_samples.legend(
        fontsize=7, framealpha=0.4, loc="upper right", labelcolor=WHITE, facecolor=BG
    )

    # 4. Grid deformation (symplectic = area-preserving)
    plot_grid(ax_grid, flow_grid, noise_grid.numpy(), lim)
    ax_grid.set_title("④ Grid Deformation  (area-preserving)", fontsize=11, pad=8)
    ax_grid.set_xlabel("q")
    ax_grid.set_ylabel("p")

    # 5. Training loss
    smooth = np.convolve(losses, np.ones(10) / 10, mode="valid")
    ax_loss.plot(losses, color=WHITE, alpha=0.2, lw=0.8)
    ax_loss.plot(smooth, color=C1, lw=1.8, label="NLL (smoothed)")
    ax_loss.set_title("⑤ Training  −log p(x)", fontsize=11, pad=8)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("NLL")
    ax_loss.legend(fontsize=7, framealpha=0.4, labelcolor=WHITE, facecolor=BG)

    # 6. Learned density (KDE contours)
    plot_density(ax_density, model, lim)
    scatter(ax_density, data_np, C1, alpha=0.15, s=3)
    ax_density.set_title("⑥ Learned Log-Density", fontsize=11, pad=8)
    ax_density.set_xlabel("q")
    ax_density.set_ylabel("p")

    fig.suptitle(
        "Symplectic Normalising Flow — Maximum Likelihood Training",
        color=WHITE,
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nFigure saved → {path}")


def make_grid_noise(n=20) -> torch.Tensor:
    g = torch.linspace(-2.5, 2.5, n)
    gq, gp = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([gq.reshape(-1), gp.reshape(-1)], dim=1)


def plot_grid(ax, flow_pts, noise_pts, lim, n=20):
    """Draw warped grid lines in the data space."""
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fp = flow_pts.reshape(n, n, 2)
    for i in range(n):
        ax.plot(fp[i, :, 0], fp[i, :, 1], color=C3, lw=0.7, alpha=0.6)
        ax.plot(fp[:, i, 0], fp[:, i, 1], color=C3, lw=0.7, alpha=0.6)


def plot_density(ax, model, lim, res=120):
    g = torch.linspace(-lim, lim, res)
    gq, gp = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([gq.reshape(-1), gp.reshape(-1)], dim=1)
    with torch.no_grad():
        logp = model.log_prob(pts).reshape(res, res).numpy()
    cmap = LinearSegmentedColormap.from_list("symp", [BG, "#1a3a4a", C1, "#ffffff"])
    ax.contourf(gq.numpy(), gp.numpy(), logp, levels=30, cmap=cmap)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, time

    os.makedirs("outputs", exist_ok=True)

    print("=" * 50)
    print("  Symplectic Normalising Flow — MLE Training")
    print("=" * 50)
    t0 = time.time()
    model, data, losses = train(
        n_data=8000,
        n_layers=12,
        hidden=64,
        epochs=600,
        batch=512,
        lr=3e-3,
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s")

    # Verify symplecticity: check det J ≈ 1 on held-out points
    x_test = sample_target(200)
    J = torch.autograd.functional.jacobian(lambda x: model.encode(x), x_test[:4])
    # jacobian shape: (4, 2, 4, 2)  — per-sample blocks on diagonal
    dets = []
    for i in range(4):
        Ji = J[i, :, i, :]  # 2x2 block for sample i
        dets.append(torch.det(Ji).item())
    print(f"\nJacobian determinants (should be ≈1): {[f'{d:.6f}' for d in dets]}")

    visualise(model, data, losses, "outputs")
