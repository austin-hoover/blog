import argparse
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


M = np.loadtxt("data/matrix.txt")

eig_res = np.linalg.eig(M)
eigvals = eig_res.eigenvalues[[0, 2]]
eigvecs = eig_res.eigenvectors[:, [0, 2]]


x = np.zeros(4)
x[0] = 10.0


turns = 1000
coords = np.zeros((turns + 1, 4))
coords[0, :] = np.copy(x)
for i in range(turns):
    x = np.matmul(M, x)
    coords[i + 1, :] = np.copy(x)
    print(x)
    
fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(coords[:, 0], coords[:, 2], s=1)
plt.show()