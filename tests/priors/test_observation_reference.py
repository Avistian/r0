"""Literal affine/activation/noise oracle for simultaneous selected-channel clamps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from r0.modules import priors as p

MATRICES = (
    ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
    ((2.0, 1.0, 0.0), (0.0, 3.0, 1.0), (1.0, 0.0, 2.0)),
    ((1.0, 2.0, 3.0), (3.0, 1.0, 2.0), (2.0, 3.0, 1.0)),
)


def recipe(
    placement: str, profile: str, target: str, multiclass: bool = False
) -> p.NetworkObservationRecipe:
    task = p.LinearRecipe(
        rows=16,
        context_rows=7,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass",
        classes=3,
    )
    source = p.NetworkRecipe(
        task,
        p.FrozenNetworkFunction(
            2,
            tuple(
                p.NetworkBlock(
                    (i,),
                    (
                        p.NetworkAffine(
                            matrix, (0.1, -0.2, 0.3), activation=p.CausalActivation("tanh")
                        ),
                    ),
                )
                for i, matrix in enumerate(MATRICES)
            ),
        ),
    )
    stage = p.NetworkInnovationStage((0.2, -0.1, 0.3), (0.1, 0.2, 0.3), placement=placement)
    noisy = p.with_network_innovations(
        source, p.NetworkInnovationPlan((stage,) * 3, importance=(1.25, 0.75))
    )
    selected = p.sample_network_observations(
        p.NetworkObservationRequest(
            noisy,
            2,
            outputs=3 if multiclass else 1,
            profile=profile,
            selection="unrestricted",
            target=target,
        ),
        rng=np.random.default_rng(9),
    )
    return p.NetworkObservationRecipe(
        replace(
            task,
            task_type="multiclass" if multiclass else "regression",
            classes=3 if multiclass else 0,
        ),
        selected,
    )


def rng(*keys: object) -> np.random.Generator:
    raw = json.dumps(("functional-causal-v1", 17, 29, keys), sort_keys=True, separators=(",", ":"))
    return np.random.default_rng(int(hashlib.sha256(raw.encode()).hexdigest(), 16))


def literal(
    r: p.NetworkObservationRecipe, x: np.ndarray, placement: str, reference: str
) -> np.ndarray:
    state = np.repeat(rng("root-base", 0).normal(size=(16, 2))[3:4], len(x), axis=0)
    outputs: list[np.ndarray] = []
    for block, matrix in enumerate(MATRICES):
        if block == 0:
            state = state * np.array([1.25, 0.75])
        affine = state @ np.array(matrix).T + np.array([0.1, -0.2, 0.3])
        node = 2 + block * (3 if placement == "pre-activation" else 2)
        noise = (
            np.array([rng("innovation", node, c).normal(size=16)[3] for c in range(3)])
            if reference == "original-row"
            else np.zeros(3)
        )
        offset = np.array([0.2, -0.1, 0.3]) + np.array([0.1, 0.2, 0.3]) * noise
        state = (
            np.tanh(affine + offset) if placement == "pre-activation" else np.tanh(affine) + offset
        )
        for column, index in enumerate(r.observations.features):
            channel = r.observations.catalog[index]
            if channel.block == block:
                state[:, channel.channel] = x[:, column]
        outputs.append(state.copy())
    return np.column_stack(
        [
            outputs[r.observations.catalog[i].block][:, r.observations.catalog[i].channel]
            for i in r.observations.targets
        ]
    )


@pytest.mark.parametrize("placement", ["pre-activation", "post-block"])
@pytest.mark.parametrize("profile", ["p1-paper-block-r0-v1", "p1-778f0d0-v1"])
@pytest.mark.parametrize("target", ["internal", "final"])
@pytest.mark.parametrize("reference", ["zero", "original-row"])
@pytest.mark.parametrize("multiclass", [False, True])
def test_literal_observation_reference(
    placement: str, profile: str, target: str, reference: str, multiclass: bool
) -> None:
    r = recipe(placement, profile, target, multiclass)
    x = torch.tensor([[-1.0, 0.3], [0.0, 0.5], [1.0, 0.7]], dtype=torch.float64)
    before = json.dumps(r.observations.raw[0].values)
    actual = p.evaluate_observation_reference(r, x, seed=17, task_id=29, row=3, reference=reference)
    np.testing.assert_allclose(
        actual.numpy(), literal(r, x.numpy(), placement, reference), rtol=0, atol=1e-12
    )
    assert json.dumps(r.observations.raw[0].values) == before
    expanded = p.evaluate_observation_reference(
        r, torch.cat((x, x[:1])), seed=17, task_id=29, row=3, reference=reference
    )
    assert torch.equal(actual, expanded[:3])


def test_anchor_identity_alias_and_target_adapter_exclusion() -> None:
    r = recipe("pre-activation", "p1-778f0d0-v1", "final")
    task = p.generate_task(r, seed=17, task_id=29)
    fixed = p.evaluate_observation_reference(r, task.inputs[3:4], seed=17, task_id=29, row=3)
    np.testing.assert_allclose(fixed.numpy(), task.scores[3:4].numpy(), rtol=0, atol=1e-12)
    altered = replace(r, task=replace(r.task, noise_ratio=12.0))
    assert torch.equal(
        fixed,
        p.evaluate_observation_reference(altered, task.inputs[3:4], seed=17, task_id=29, row=3),
    )


def test_reference_admission_precedes_finite_masks_and_snapshot_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = recipe("post-block", "p1-778f0d0-v1", "final")
    x = torch.zeros((20000, 2), dtype=torch.float64)

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("allocation or custody replay preceded aggregate admission")

    # The valid source remains constructible under 8 MiB; the requested population does not.
    source = r.observations.request.source
    assert isinstance(source, p.NetworkInnovationRecipe)
    assert source.source is not None
    assert source.plan is not None
    bounded_source = p.with_network_innovations(
        replace(source.source, limits=p.CausalLimits(working_bytes=8 * 1024**2)), source.plan
    )
    selected = p.sample_network_observations(
        replace(r.observations.request, source=bounded_source), rng=np.random.default_rng(9)
    )
    bounded = p.NetworkObservationRecipe(r.task, selected)
    monkeypatch.setattr(torch, "isfinite", forbidden)
    monkeypatch.setattr(type(selected.initial), "restore", forbidden)
    with pytest.raises(ValueError, match="limit|working"):
        p.evaluate_observation_reference(bounded, x, seed=17, task_id=29, row=3)


@pytest.mark.parametrize("bad", ["nan", "dtype", "empty", "seed", "row", "mode"])
def test_reference_rejects_invalid_inputs(bad: str) -> None:
    r = recipe("post-block", "p1-778f0d0-v1", "final")
    x = torch.zeros((2, 2), dtype=torch.float64)
    if bad == "nan":
        x[0, 0] = float("nan")
    elif bad == "dtype":
        x = x.float()
    elif bad == "empty":
        x = x[:0]
    with pytest.raises(ValueError, match="reference"):
        p.evaluate_observation_reference(
            r,
            x,
            seed=True if bad == "seed" else 17,
            task_id=29,
            row=16 if bad == "row" else 3,
            reference="invalid" if bad == "mode" else "zero",
        )
