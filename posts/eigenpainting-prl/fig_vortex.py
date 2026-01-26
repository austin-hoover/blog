import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True


levels = 5
radii = np.linspace(0.0, 1.0, levels + 1)

points = []
for r in radii[1:]:
    # m = np.exp(r) * 7
    m = 5 + int(r * 10)
    # m = 15
    m = int(m)

    t = np.linspace(0.0, 2.0 * np.pi, m)

    points_new = np.zeros((m, 4))
    points_new[:, 0] = np.cos(t)
    points_new[:, 2] = np.sin(t)
    points_new[:, 1] = -points_new[:, 2]
    points_new[:, 3] = +points_new[:, 0]
    points_new *= r
    points.append(points_new)
points = np.vstack(points)


points1 = np.copy(points)

points2 = np.copy(points)
scale = 1.25
points2[:, 0] *= scale
points2[:, 1] *= scale
points2[:, 2] /= scale
points2[:, 3] /= scale

points3 = np.copy(points2)
phi = np.pi * 0.25
c = np.cos(phi)
s = np.sin(phi)
R = np.array([
    [c, 0, s, 0],
    [0, c, 0, s],
    [-s, 0, c, 0],
    [0, -s, 0, c],
])
points3 = np.matmul(points3, R.T)

points = points3

fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, constrained_layout=True, figsize=(8, 2.5))
xmax = 2.0
for ax, points in zip(axs, [points1, points2, points3]):
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)
    ax.quiver(points[:, 0], points[:, 2], points[:, 1], points[:, 3])
for ax in axs.flat:
    ax.axis("off")
    ax.set_aspect(1.0)
plt.savefig("images/vortex.png", dpi=300)