from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .complex import CubicalTumourComplex


def solve_transport_numpy(complex_: CubicalTumourComplex, conductance: np.ndarray,
                          reaction_rate: float = 0.045,
                          penalty: float = 250.0) -> np.ndarray:
    g = np.asarray(conductance, dtype=float)
    if g.shape != (complex_.n_edges,):
        raise ValueError("conductance has wrong shape")
    if np.any(g <= 0):
        raise ValueError("conductance must be positive")
    D = complex_.D
    A = D.T @ sp.diags(g) @ D + reaction_rate * sp.eye(complex_.n_cells, format="csr")
    src = complex_.source_mask.astype(float)
    A = A + penalty * sp.diags(src)
    b = penalty * src
    c = spla.spsolve(A, b)
    return np.clip(c, 0.0, 1.05)


def solve_transport_torch(D: torch.Tensor, conductance: torch.Tensor,
                          source_mask: torch.Tensor, reaction_rate: float,
                          penalty: float) -> torch.Tensor:
    """Batched differentiable CMC transport solve.

    D: [E,N], conductance: [B,E], source_mask: [N].
    """
    if conductance.ndim == 1:
        conductance = conductance.unsqueeze(0)
    B, _ = conductance.shape
    Dt = D.transpose(0, 1)
    weighted_D = conductance.unsqueeze(-1) * D.unsqueeze(0)
    A = torch.matmul(Dt.unsqueeze(0), weighted_D)
    eye = torch.eye(D.shape[1], dtype=D.dtype, device=D.device).expand(B, -1, -1)
    A = A + reaction_rate * eye
    src_diag = torch.diag(source_mask).unsqueeze(0)
    A = A + penalty * src_diag
    b = penalty * source_mask.unsqueeze(0).expand(B, -1)
    return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
