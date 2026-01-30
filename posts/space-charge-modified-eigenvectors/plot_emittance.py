import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["figure.constrained_layout.use"] = True

colors = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']

histories = [
    pd.read_csv("data/run_01/history.dat", index_col=0),
    pd.read_csv("data/run_02/history.dat", index_col=0),
    pd.read_csv("data/run_03/history.dat", index_col=0)
]

plot_kws = dict(
    lw=1.5,
    ls="-",
)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.plot(histories[0]["eps_1"] * 1e6, color=colors[0], alpha=0.15, label=r"$\varepsilon_1 (I = 0)$", **plot_kws)
ax.plot(histories[1]["eps_1"] * 1e6, color=colors[0], alpha=0.5, label=r"$\varepsilon_1 (I = I_0 / 2)$", **plot_kws)
ax.plot(histories[2]["eps_1"] * 1e6, color=colors[0], alpha=1.0, label=r"$\varepsilon_1 (I = I_0)$", **plot_kws)
ax.plot(histories[0]["eps_2"] * 1e6, color=colors[1], alpha=0.15, label=r"$\varepsilon_2 (I = 0)$", **plot_kws)
ax.plot(histories[1]["eps_2"] * 1e6, color=colors[1], alpha=0.5, label=r"$\varepsilon_2 (I = I_0 / 2)$", **plot_kws)
ax.plot(histories[2]["eps_2"] * 1e6, color=colors[1], alpha=1.0, label=r"$\varepsilon_2 (I = I_0)$", **plot_kws)
ax.set_xlabel("Turn")
ax.set_ylabel("[mm mrad]")
ax.legend()
plt.savefig("images/scale_intensity_eps2.png", dpi=300)