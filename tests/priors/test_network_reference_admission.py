"""Actual finite-mask allocation must follow reference workspace admission."""

import pytest
import torch

from r0.modules import priors as p


def reference() -> p.NetworkInnovationRecipe:
    source = p.NetworkRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        p.FrozenNetworkFunction(1, (p.NetworkBlock((0,), (p.NetworkAffine(((1.0,),), (0.1,)),)),)),
        limits=p.CausalLimits(working_bytes=1024**2),
    )
    return p.with_network_innovations(
        source, p.NetworkInnovationPlan((p.NetworkInnovationStage((0.0,), (0.2,)),))
    )


def test_reference_admits_before_actual_finite_mask(monkeypatch: pytest.MonkeyPatch):
    recipe = reference()
    inputs = torch.zeros((1_100_000, 1), dtype=torch.float64)
    original = torch.isfinite
    mask_bytes: list[int] = []

    def observe(value: torch.Tensor) -> torch.Tensor:
        result = original(value)
        if value is inputs:
            mask_bytes.append(result.numel() * result.element_size())
        return result

    monkeypatch.setattr(torch, "isfinite", observe)
    with pytest.raises(ValueError, match="working memory"):
        p.evaluate_network_innovation_reference(recipe, inputs)
    assert mask_bytes == []


def test_admitted_nonfinite_reference_still_rejects():
    with pytest.raises(ValueError, match="finite observed"):
        p.evaluate_network_innovation_reference(
            reference(), torch.tensor([[float("nan")]], dtype=torch.float64)
        )
