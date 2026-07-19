from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from cmc_tumour.complex import CubicalTumourComplex
from cmc_tumour.phantom import generate_sample
from cmc_tumour.transport import solve_transport_numpy


def test_incidence_cancels_internal_flux() -> None:
    complex_ = CubicalTumourComplex.circular(9)
    assert np.allclose(complex_.D @ np.ones(complex_.n_cells), 0.0)


def test_positive_conductance_and_bounded_solution() -> None:
    complex_ = CubicalTumourComplex.circular(9)
    sample = generate_sample(complex_, 2026)
    assert np.all(sample["conductance"] > 0)
    c = solve_transport_numpy(complex_, sample["conductance"])
    assert np.all(np.isfinite(c))
    assert c.min() >= 0.0
    assert c.max() <= 1.05


def test_interior_balance_is_machine_precision() -> None:
    complex_ = CubicalTumourComplex.circular(9)
    sample = generate_sample(complex_, 2027)
    c = solve_transport_numpy(complex_, sample["conductance"])
    flux = sample["conductance"] * (complex_.D @ c)
    balance = complex_.D.T @ flux + 0.045 * c
    assert np.max(np.abs(balance[~complex_.source_mask])) < 1e-10
