import copy

import numpy as np

from orbit.core.bunch import Bunch
from orbit.core.spacecharge import Grid1D
from orbit.core.spacecharge import Grid2D


def get_grid_points(grid_coords: list[np.ndarray]) -> np.ndarray:
    if len(grid_coords) == 1:
        return grid_coords[0]
    return np.vstack([c.ravel() for c in np.meshgrid(*grid_coords, indexing="ij")]).T


def grid_edges_to_coords(grid_edges: np.ndarray) -> np.ndarray:
    return 0.5 * (grid_edges[:-1] + grid_edges[1:])


def grid_coords_to_edges(grid_coords: np.ndarray) -> np.ndarray:
    delta = grid_coords[1] - grid_coords[0]
    grid_edges = np.zeros(grid_coords.shape[0] + 1)
    grid_edges[0] = grid_coords[0] - 0.5 * delta
    grid_edges[1:] = grid_coords + 0.5 * delta
    return grid_edges


def make_grid(shape: tuple[int, ...], limits: list[tuple[float, float]]) -> Grid2D:
    if len(shape) == 1:
        return Grid1D(shape[0], limits[0][0], limits[0][1])
    elif len(shape) == 2:
        return Grid2D(
            shape[0] + 1,
            shape[1] + 1,
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1],
        )
    else:
        raise ValueError


class Histogram:
    def __init__(self, values: np.ndarray, edges: np.ndarray = None, coords: np.ndarray = None) -> None:
        self.values = np.copy(values)

        self.coords = coords
        self.edges = edges
        if self.coords is None and self.edges is not None:
            self.coords = [grid_edges_to_coords(e) for e in self.edges]
        if self.edges is None and self.coords is not None:
            self.edges = [grid_coords_to_edges(c) for c in self.coords]

        self.coords = [np.copy(_) for _ in self.coords]
        self.edges = [np.copy(_) for _ in self.edges]

    def copy(self):
        return copy.deepcopy(self)


class BunchHistCalc:
    def __init__(
        self,
        axis: tuple[int, ...],
        shape: tuple[int, ...],
        limits: list[tuple[float, float]],
    ) -> None:
        self.axis = axis
        self.ndim = len(axis)

        self.dims = ["x", "xp", "y", "yp", "z", "dE"]
        self.dims = [self.dims[i] for i in self.axis]

        self.grid_shape = shape
        self.grid_limits = limits
        self.grid_edges = [
            np.linspace(self.grid_limits[i][0], self.grid_limits[i][1], self.grid_shape[i] + 1)
            for i in range(self.ndim)
        ]
        self.grid_coords = [grid_edges_to_coords(e) for e in self.grid_edges]
        self.grid_values = np.zeros(shape)
        self.grid_points = get_grid_points(self.grid_coords)
        self.grid = make_grid(self.grid_shape, self.grid_limits)

        self.inv_cell_volume = np.prod([e[1] - e[0] for e in self.grid_edges])

    def bin_bunch(self, bunch: Bunch) -> None:
        macrosize = bunch.macroSize()
        if macrosize == 0:
            bunch.macroSize(1.0)
        
        self.grid.binBunch(bunch, *self.axis)

        bunch.macroSize(macrosize)

    def compute_histogram(self, bunch: Bunch) -> np.ndarray:
        self.bin_bunch(bunch)

        values = np.zeros(self.grid_points.shape[0])
        for i, indices in enumerate(np.ndindex(*self.grid_shape)):
            values[i] = self.grid.getValueOnGrid(*indices)
        return values.reshape(self.grid_shape)

    def __call__(self, bunch: Bunch) -> Histogram:
        self.grid.setZero()
        self.grid_values = self.compute_histogram(bunch)
        return Histogram(values=self.grid_values, edges=self.grid_edges)
