from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PositiveEdgeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.012 + F.softplus(self.net(x).squeeze(-1))


class HodgeEdgeNet(nn.Module):
    """Positive edge law enriched by fixed lower-Hodge message passing.

    The edge 1-cochain features are propagated by H and H^2 before a shared
    constitutive network is evaluated. This is a compact cell-complex neural
    operator: adjacent interfaces communicate only through incidence-derived
    Hodge structure, not through Euclidean nearest-neighbour searches.
    """

    def __init__(self, in_dim: int, hidden: int = 24, layers: int = 2):
        super().__init__()
        del layers  # retained for configuration compatibility
        self.net = nn.Sequential(
            nn.Linear(3 * in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        h1 = torch.matmul(H, x)
        h2 = torch.matmul(H, h1)
        z = torch.cat([x, h1, h2], dim=-1)
        return 0.012 + F.softplus(self.net(z).squeeze(-1))
