"""Vector clamps replace logical emissions without splitting the joint mechanism."""

import torch

from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.native_vector_intervention import NativeVectorHardIntervention
from r0.modules.priors.lib.native_vector_records import NativeVectorNode


def vector_clamps(execution: InterventionExecution | None, index: int):
    return (
        {}
        if execution is None or execution.baseline
        else {
            c.variable: c
            for c in execution.recipe.interventions
            if isinstance(c, NativeVectorHardIntervention) and c.node == index
        }
    )


def forced_variable(
    change: NativeVectorHardIntervention, node: NativeVectorNode, rows: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    domain = node.domains[change.variable]
    span = node.spans[change.variable]
    if domain.kind == "continuous":
        assert change.value is not None
        value = torch.full((rows,), change.value, dtype=torch.float64, device=device)
        return value, value[:, None], value[:, None]
    category = domain.vocabulary.index(change.category)
    value = torch.full((rows,), category, dtype=torch.int64, device=device)
    encoded = torch.nn.functional.one_hot(value, len(domain.vocabulary)).to(torch.float64)
    # Threshold mechanisms have one score slot. It now carries a forced category index,
    # explicitly distinguished from the original threshold score and K-wide probability.
    score = value[:, None].to(torch.float64) if span.score_stop - span.score_start == 1 else encoded
    return value, encoded, score
