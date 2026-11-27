"""Explicit asymmetric concentrations retain Gamma draws and native replay custody."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.generation import load_group, save_group


def prior(**kwargs: object) -> p.AsymmetricNativeDomainPrior:
    fields: dict[str, Any] = dict(
        kinds=("categorical",),
        cardinalities=(3,),
        emissions=("softmax",),
        concentration_vectors=((0.5, 2.0, 7.0),),
    )
    fields.update(kwargs)
    return p.AsymmetricNativeDomainPrior(**fields)


def request(domain: p.NativeDomainPrior, parent: bool = False) -> p.NativeGraphRequest:
    kind = p.NativeParentGraphRequest if parent else p.NativeGraphRequest
    return kind(
        parents=((), (0,)),
        domains=(domain, domain),
        observed=((0, 0),),
        target=(1, 0),
        rows=16,
        context_rows=8,
        calibration_rows=8,
        root_prior=p.JointRootPrior(
            families=("gaussian",), function=p.JointFunctionPrior(functions="lin")
        ),
        function_prior=p.JointFunctionPrior(functions="lin"),
    )


def sample(r: p.NativeGraphRequest, seed: int = 1):
    return p.sample_native_graph(
        r,
        **{
            k: np.random.default_rng(seed + i)
            for i, k in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
        },
    )


@pytest.mark.parametrize("parent", [False, True])
@pytest.mark.parametrize("emission", ["softmax", "argmax", "threshold"])
def test_absolute_alpha_formula_replay_and_typed_archive(
    parent: bool, emission: str, tmp_path: Path
) -> None:
    graph = sample(
        request(prior(emissions=(emission,), threshold_laws=("uniform-quantile",)), parent)
    )
    # NumPy's independent Dirichlet API is the distribution oracle for the first raw draws.
    first = graph.recipe.nodes[0].domain
    assert isinstance(first, p.FrequencyVariableDomain)
    np.testing.assert_allclose(
        first.frequencies, np.random.default_rng(2).dirichlet([0.5, 2.0, 7.0]), rtol=1e-15
    )
    for plan, node in zip(graph.domains, graph.recipe.nodes, strict=True):
        gamma = [
            d for d in graph.raw if d.owner == plan.operation and d.role.startswith("frequency:")
        ]
        assert len(gamma) == 3
        assert [dict(d.parameters)["concentration"] for d in gamma] == [0.5, 2.0, 7.0]
        assert isinstance(node.domain, p.FrequencyVariableDomain)
        values = np.array([d.values[0] for d in gamma])
        np.testing.assert_array_equal(node.domain.frequencies, values / values.sum())
        if emission == "threshold":
            np.testing.assert_allclose(
                node.domain.thresholds,
                (2 * np.cumsum(values / values.sum())[:-1] - 1) * plan.threshold_scale,
            )
    assert not any(e.role == "concentration" for e in graph.registry.decisions.events)
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(graph)))) == graph
    task = p.generate_task(graph, seed=8, task_id=9)
    assert p.explain_generation(task, [3, 1, 3])["rows"] == [3, 1, 3]
    # Authored equations carry the same realized domain without a new evaluator version.
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(graph.recipe)))) == graph.recipe
    save_group(tmp_path / "asymmetric.zip", [(9, task)], max_bytes=5_000_000)
    assert next(load_group(tmp_path / "asymmetric.zip")).episode() == task.episode()
    altered = json.loads(json.dumps(asdict(graph)))
    altered["domains"][0]["concentrations"][0] += 1
    with pytest.raises(ValueError, match="replay"):
        p.recipe_from_dict(altered)


@pytest.mark.parametrize(
    "vectors",
    [
        (),
        ((1.0, 2.0),),
        ((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)),
        ((1.0, 0.0, 2.0),),
        ((1.0, float("nan"), 2.0),),
        ((True, 1.0, 2.0),),
    ],
)
def test_vector_cardinality_and_positive_contract(vectors: tuple[tuple[float, ...], ...]) -> None:
    with pytest.raises(ValueError, match="vector"):
        prior(concentration_vectors=vectors)


def test_cardinalities_mixed_laws_and_four_stream_rollback() -> None:
    d = prior(
        kinds=("binary", "categorical"),
        cardinalities=(3, 4),
        concentration_vectors=((1.0, 3.0), (0.5, 2.0, 7.0), (1.0, 2.0, 3.0, 4.0)),
        frequency_laws=("uniform", "dirichlet", "logistic-normal", "asymmetric-dirichlet"),
    )
    seen: set[str] = set()
    for seed in range(15):
        g = sample(request(d), seed)
        seen.update(plan.frequency_law for plan in g.domains)
        assert p.recipe_from_dict(json.loads(json.dumps(asdict(g)))) == g
        if len(seen) == 4:
            break
    assert len(seen) == 4
    for bad in (
        request(prior(concentration_vectors=((1e-300, 1e-300, 1e-300),))),
        replace(request(prior()), raw_limits=p.RawDrawLimits(draws=1)),
    ):
        streams = {
            k: np.random.default_rng(i)
            for i, k in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
        }
        before = json.dumps([r.bit_generator.state for r in streams.values()])
        with pytest.raises(ValueError, match="underflow|limit"):
            p.sample_native_graph(bad, **streams)
        assert json.dumps([r.bit_generator.state for r in streams.values()]) == before


@pytest.mark.parametrize("mode", ["local", "global", "meta"])
def test_asymmetric_named_scope_and_absolute_alpha_contract(mode: str) -> None:
    domain = prior(
        cardinalities=(3, 4), concentration_vectors=((1.0, 2.0, 7.0), (1.0, 2.0, 3.0, 4.0))
    )
    graph = sample(replace(request(domain), mode=mode))
    choices = [
        d
        for d in graph.registry.decisions.categorical.draws
        if d.request.name == "native_cardinality"
    ]
    assert len(choices) == 2
    assert choices[0].preference == choices[1].preference
    if mode == "global":
        assert graph.domains[0].cardinality == graph.domains[1].cardinality
    changed = sample(replace(request(replace(domain, concentration=(100.0, 100.0))), mode=mode))
    assert graph.raw == changed.raw
    assert tuple(n.domain for n in graph.recipe.nodes) == tuple(
        n.domain for n in changed.recipe.nodes
    )
