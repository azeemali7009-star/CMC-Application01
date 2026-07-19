from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import spsolve

from .complex import CubicalTumourComplex


def _sample_field(complex_: CubicalTumourComplex, rng: np.random.Generator) -> np.ndarray:
    n = complex_.mask.shape[0]
    raw = rng.normal(size=(n, n))
    smooth = gaussian_filter(raw, sigma=rng.uniform(1.0, 2.2))
    vals = smooth[complex_.mask]
    vals = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-12)
    return vals


def generate_sample(complex_: CubicalTumourComplex, seed: int) -> dict[str, np.ndarray]:
    """Generate one heterogeneous tumour transport realisation.

    The hidden conductance law includes a non-local Hodge-smoothed stromal
    contribution. This creates a controlled benchmark in which independent
    edge regression is structurally misspecified while a flux-cochain model
    can exploit edge-edge incidence.
    """
    rng = np.random.default_rng(seed)
    xy = complex_.coords
    stroma = _sample_field(complex_, rng)
    necrosis = np.exp(-((xy[:, 0] - 0.58) ** 2 + (xy[:, 1] - 0.52) ** 2) /
                       rng.uniform(0.025, 0.055))
    necrosis = np.clip(necrosis + 0.18 * _sample_field(complex_, rng), 0.0, 1.0)
    endothelial = np.exp(-((xy[:, 0] - 0.28) ** 2 + (xy[:, 1] - 0.34) ** 2) / 0.028)
    endothelial += 0.7 * np.exp(-((xy[:, 0] - 0.22) ** 2 + (xy[:, 1] - 0.70) ** 2) / 0.018)
    endothelial = np.clip(endothelial, 0.0, 1.0)
    radial = np.sqrt((xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2)
    degree = complex_.degree / max(complex_.degree.max(), 1.0)
    cell_features = np.column_stack([stroma, necrosis, endothelial, radial, degree, xy])

    edge_features = complex_.edge_features_from_cells(cell_features)
    u, v = complex_.edges[:, 0], complex_.edges[:, 1]
    edge_stroma = 0.5 * (stroma[u] + stroma[v])
    edge_necrosis = 0.5 * (necrosis[u] + necrosis[v])
    edge_endo = 0.5 * (endothelial[u] + endothelial[v])

    # Curved stromal barrier with sparse gates.
    mid = 0.5 * (xy[u] + xy[v])
    barrier_curve = np.abs(mid[:, 0] - (0.52 + 0.10 * np.sin(7.0 * mid[:, 1])))
    gates = (np.abs(mid[:, 1] - 0.30) < 0.06) | (np.abs(mid[:, 1] - 0.72) < 0.05)
    barrier = ((barrier_curve < 0.045) & (~gates)).astype(float)

    H = complex_.edge_hodge()
    # A controlled non-local constitutive component generated exclusively from
    # incidence-derived edge coupling.  HodgeEdgeNet receives the operators H
    # and H^2; an independent edge MLP does not.
    hodge_context = H @ edge_stroma + 0.80 * (H @ (H @ edge_stroma))
    hodge_context = (hodge_context - hodge_context.mean()) / max(
        hodge_context.std(), 1e-8
    )

    log_g = (
        -0.30
        - 0.85 * edge_stroma
        - 0.85 * edge_necrosis
        - 1.30 * barrier
        + 0.55 * edge_endo
        + 1.25 * hodge_context
        + 0.10 * rng.normal(size=complex_.n_edges)
    )
    conductance = np.clip(np.exp(log_g), 0.015, 3.0)
    return {
        "cell_features": cell_features.astype(np.float32),
        "edge_features": edge_features.astype(np.float32),
        "conductance": conductance.astype(np.float32),
        "barrier": barrier.astype(np.float32),
        "stroma": stroma.astype(np.float32),
        "necrosis": necrosis.astype(np.float32),
        "endothelial": endothelial.astype(np.float32),
    }
