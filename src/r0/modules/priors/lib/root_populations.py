"""Explicit external root rows, separate from iid root laws and mechanisms."""

import math
from dataclasses import dataclass
from typing import Any

import torch

from r0.modules.priors.lib.weight_geometry import Matrix


@dataclass(frozen=True, slots=True)
class RootPopulations:
    """An enclosing admitted recipe owns matrices in exact original-row order."""

    populations: tuple[tuple[int, Matrix], ...]
    population_id: str
    row_ids: tuple[int, ...]
    source_columns: tuple[tuple[int, tuple[int, ...]], ...]

    def values(self, node: int, rows: int, width: int, device: str) -> torch.Tensor | None:
        matrix = next((values for index, values in self.populations if index == node), None)
        if matrix is None:
            return None
        if rows != len(self.row_ids) or len(matrix) != rows or any(len(r) != width for r in matrix):
            raise ValueError("external root population requires its complete original row geometry")
        return torch.tensor(matrix, dtype=torch.float64, device=device)

    def metadata(self, node: int) -> dict[str, Any]:
        return {"population_id": self.population_id, "node": node, "row_ids": self.row_ids}

    def certificates(self, node: int):
        from r0.modules.priors.lib.functional_analysis import ChannelCertificate

        columns = next((columns for index, columns in self.source_columns if index == node), None)
        if columns is None:
            return None
        # Negative namespace cannot collide with ordinary nonnegative graph node IDs.
        # Reusing one GP column at different roots retains its shared causal source.
        return [
            ChannelCertificate(frozenset({(-1, column)}), math.inf, False) for column in columns
        ]
