from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class CubicalTumourComplex:
    """Cell-centred 2D cubical tumour complex.

    The active pixels are physical 2-cells. Shared pixel faces are oriented
    transport 1-cells. ``D`` is the oriented interface-to-cell incidence
    matrix. Consequently, ``D.T @ j`` is the conservative cell balance and
    every interior flux appears with opposite signs in its two incident cells.
    """

    mask: np.ndarray
    coords: np.ndarray
    edges: np.ndarray
    D: sp.csr_matrix
    source_mask: np.ndarray
    degree: np.ndarray

    @classmethod
    def circular(cls, n: int = 17) -> "CubicalTumourComplex":
        if n < 7:
            raise ValueError("n must be at least 7")
        yy, xx = np.mgrid[0:n, 0:n]
        cx = cy = (n - 1) / 2
        radius = 0.47 * n
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        index = -np.ones_like(mask, dtype=int)
        coords_list: list[tuple[float, float]] = []
        for y, x in zip(*np.nonzero(mask), strict=True):
            index[y, x] = len(coords_list)
            coords_list.append((x / (n - 1), y / (n - 1)))

        oriented_edges: list[tuple[int, int]] = []
        for y, x in zip(*np.nonzero(mask), strict=True):
            i = index[y, x]
            for dy, dx in ((0, 1), (1, 0)):
                yn, xn = y + dy, x + dx
                if 0 <= yn < n and 0 <= xn < n and mask[yn, xn]:
                    oriented_edges.append((i, index[yn, xn]))

        edges = np.asarray(oriented_edges, dtype=int)
        n_cells = len(coords_list)
        n_edges = len(edges)
        rows = np.repeat(np.arange(n_edges), 2)
        cols = edges.reshape(-1)
        vals = np.tile(np.array([-1.0, 1.0]), n_edges)
        D = sp.coo_matrix((vals, (rows, cols)), shape=(n_edges, n_cells)).tocsr()
        degree = np.asarray(np.abs(D).sum(axis=0)).ravel()

        coords = np.asarray(coords_list, dtype=float)
        # Vascular/source cells: a left boundary arc plus one internal vessel.
        source_mask = (coords[:, 0] < 0.12) & (np.abs(coords[:, 1] - 0.5) < 0.28)
        source_mask |= ((coords[:, 0] - 0.38) ** 2 + (coords[:, 1] - 0.35) ** 2 < 0.022**2)
        if not np.any(source_mask):
            source_mask[np.argmin(coords[:, 0])] = True
        return cls(mask=mask, coords=coords, edges=edges, D=D,
                   source_mask=source_mask, degree=degree)

    @property
    def n_cells(self) -> int:
        return self.coords.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]

    def edge_hodge(self) -> np.ndarray:
        """Normalised lower Hodge operator on interface 1-cochains."""
        L1 = (self.D @ self.D.T).toarray().astype(np.float64)
        scale = np.maximum(np.sum(np.abs(L1), axis=1), 1.0)
        return L1 / scale[:, None]

    def edge_features_from_cells(self, cell_features: np.ndarray) -> np.ndarray:
        u, v = self.edges[:, 0], self.edges[:, 1]
        avg = 0.5 * (cell_features[u] + cell_features[v])
        diff = np.abs(cell_features[u] - cell_features[v])
        return np.concatenate([avg, diff], axis=1)
