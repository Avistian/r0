"""Sampled policies retain source selection, declared laws and complete replay."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from r0.modules import priors as p


def graph():
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, active_features=1, signal_features=1)
    nodes = (
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
        ),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),))),
        p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((3.0,),))),
    )
    return p.FunctionalCausalRecipe(task, nodes, ((0, 0),), ((2, 0),))


def test_source_policy_retains_hard_population_and_exact_archive():
    request = p.InterventionPolicyRequest(graph(), label_nodes=(2,), kind="hard")
    sampled = p.sample_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    assert len(sampled.selected) == 1
    assert sampled.selected[0] != 2
    assert sampled.runtime.mode == "population"
    assert p.recipe_from_dict(asdict(sampled)) == sampled
    task = p.generate_task(sampled, seed=11, task_id=13)
    assert p.explain_generation(task, [15, 0, 15])["rows"] == [15, 0, 15]


def test_soft_parameter_laws_all_declared_families():
    from dataclasses import replace

    laws: tuple[tuple[str, dict[str, Any]], ...] = (
        ("normal-r0", {}),
        ("uniform-r0", {"location": -2.0, "scale": 4.0}),
        ("log-uniform-r0", {"location": 0.1, "scale": 3.0}),
        ("choice-r0", {"choices": (-2.0, 3.0)}),
    )
    for family, kwargs in laws:
        functions = (
            (p.FrozenLinearFunction(((2.0,),)), ("matrix",)),
            (p.FrozenTreeFunction(1, ((0,),), ((0.0,),), (((1.0,), (2.0,)),)), ("leaves",)),
            (p.FrozenRFFFunction(((1.0, 2.0),), (0.0, 1.0), ((1.0,), (2.0,))), ("coefficients",)),
            (
                p.FrozenNetworkFunction(
                    1, (p.NetworkBlock((0,), (p.NetworkAffine(((1.0,),), (0.0,)),)),)
                ),
                ("blocks", "0", "experts", "0", "matrix"),
            ),
        )
        for function, path in functions:
            base = graph()
            base = replace(
                base,
                nodes=(base.nodes[0], replace(base.nodes[1], function=function), base.nodes[2]),
            )
            law = p.InterventionParameterLaw(1, path, family, **kwargs)
            request = p.InterventionPolicyRequest(
                base,
                (2,),
                kind="soft",
                profile="explicit-policy-r0-v1",
                eligibility=(1,),
                count=1,
                mode="matched",
                parameter_laws=(law,),
            )
            sampled = p.sample_intervention_policy(
                request,
                selection_rng=np.random.default_rng(11),
                value_rng=np.random.default_rng(13),
                calibration_rng=np.random.default_rng(17),
            )
            change = sampled.runtime.interventions[0]
            assert isinstance(change, p.SoftIntervention)
            assert type(change.function) is type(function)
            assert change.function is not None
            actual: Any = asdict(change.function)
            for token in path:
                actual = actual[int(token)] if type(actual) in (list, tuple) else actual[token]
            actual_values = np.asarray(actual, dtype=np.float64).reshape(-1)
            oracle = np.random.default_rng(13)
            if family == "normal-r0":
                expected = oracle.normal(size=len(actual_values))
            else:
                quantiles = oracle.random(len(actual_values))
                if family == "uniform-r0":
                    expected = -2.0 + 4.0 * quantiles
                elif family == "log-uniform-r0":
                    expected = np.exp((1 - quantiles) * np.log(0.1) + quantiles * np.log(3.0))
                else:
                    expected = np.where(quantiles < 0.5, -2.0, 3.0)
            np.testing.assert_allclose(actual_values, expected, rtol=1e-14, atol=1e-14)
            assert sampled.bounds == ()
            assert sampled.reference_raw == ()
            assert sampled.initial[2] == sampled.final[2]
            assert p.recipe_from_dict(asdict(sampled)) == sampled
            pair = p.generate_policy_pair(sampled, seed=19, task_id=23)
            assert pair.before.recipe_json != pair.after.recipe_json
            assert (
                p.explain_generation(pair.after, [3, 0, 3])["episode_id"] == pair.after.episode_id
            )


def test_policy_failure_atomic_and_strict_custody():
    import copy
    from dataclasses import replace
    from unittest.mock import patch

    import pytest

    request = p.InterventionPolicyRequest(graph(), (2,))
    streams = [np.random.default_rng(k) for k in (3, 5, 7)]
    before = [copy.deepcopy(r.bit_generator.state) for r in streams]
    with (
        patch("copy.deepcopy", side_effect=AssertionError("snapshot before admission")),
        pytest.raises(ValueError, match="admission|memory|working|limit"),
    ):
        p.sample_intervention_policy(
            replace(request, raw_limits=p.RawDrawLimits(numbers=1)),
            selection_rng=streams[0],
            value_rng=streams[1],
            calibration_rng=streams[2],
        )
    assert before == [r.bit_generator.state for r in streams]
    sampled = p.sample_intervention_policy(
        request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
    )
    data = asdict(sampled)
    for key in data:
        malformed = copy.deepcopy(data)
        del malformed[key]
        with pytest.raises(ValueError, match="policy|archive"):
            p.recipe_from_dict(malformed)
    malformed = copy.deepcopy(data)
    malformed["selected"] = (False,)
    with pytest.raises(ValueError, match="policy|replay"):
        p.recipe_from_dict(malformed)
    task = p.generate_task(sampled, seed=19, task_id=23)
    with pytest.raises(ValueError, match="semantic"):
        p.explain_generation(replace(task, task_type="binary"), [0])


def test_keyed_reference_oracle_five_engines_and_source_bounds():
    import hashlib
    import json
    from dataclasses import replace

    for engine in (
        np.random.PCG64,
        np.random.PCG64DXSM,
        np.random.Philox,
        np.random.SFC64,
        np.random.MT19937,
    ):
        for lifecycle in ("independent-calibration-r0", "source-reference-population"):
            streams = [np.random.Generator(engine(i)) for i in (3, 5, 7)]
            oracle = [np.random.Generator(engine(i)) for i in (3, 5, 7)]
            request = p.InterventionPolicyRequest(
                graph(),
                (2,),
                bounds_lifecycle=lifecycle,
                reference_rows=16,
                reference_seed=29,
                reference_task_id=31,
            )
            sampled = p.sample_intervention_policy(
                request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
            )
            selected = int(oracle[0].permutation(2)[0])
            assert sampled.selected == (selected,)
            seed = int(oracle[2].integers(2**63 - 1)) if lifecycle.startswith("independent") else 29
            keys = (
                ("policy-independent-reference-v1", "root-base", 0)
                if lifecycle.startswith("independent")
                else ("root-base", 0)
            )
            encoded = json.dumps(
                ("functional-causal-v1", seed, 31, keys), sort_keys=True, separators=(",", ":")
            )
            root = np.random.default_rng(
                int(hashlib.sha256(encoded.encode()).hexdigest(), 16)
            ).normal(size=(16, 1))
            values = root * (2 if selected else 1)
            low, high = values.min(), values.max()
            u = oracle[1].random(1)[0]
            change = sampled.runtime.interventions[0]
            assert isinstance(change, p.HardIntervention)
            np.testing.assert_array_equal(change.values, [(1 - u) * low + u * high])
            np.testing.assert_array_equal(sampled.bounds[0].minimum, [low])
            for actual, expected in zip(streams, oracle, strict=True):
                np.testing.assert_array_equal(actual.random(5), expected.random(5))
            altered = replace(request.base.task, context_rows=7)
            assert replace(request.base, task=altered).nodes == request.base.nodes
            sampled.validate_replay()


def test_large_source_count_and_matched_vector_pipeline():
    from dataclasses import replace

    base = graph()
    nodes = (base.nodes[0],) + tuple(
        p.FunctionalCausalNode((i - 1,), p.FrozenLinearFunction(((1.0,),))) for i in range(1, 22)
    )
    base = replace(base, nodes=nodes, target=((21, 0),))
    request = p.InterventionPolicyRequest(base, (21,), reference_rows=16)
    assert request.selected_count == 2
    sampled = p.sample_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    assert len(set(sampled.selected)) == 2
    assert 21 not in sampled.selected
    root = p.FunctionalCausalNode(
        (),
        p.FrozenLinearFunction(((1.0,), (2.0,))),
        p.JointRootBase("gaussian", 1),
        pipeline=p.CausalNodePipeline(channels=2, node_importance=3.0),
    )
    end = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, -1.0),)))
    vector = replace(graph(), nodes=(root, end), target=((1, 0),))
    req = p.InterventionPolicyRequest(
        vector, (1,), profile="explicit-policy-r0-v1", count=1, mode="matched", reference_rows=16
    )
    sampled = p.sample_intervention_policy(
        req,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    pair = p.generate_policy_pair(sampled, seed=19, task_id=23)
    change = sampled.runtime.interventions[0]
    assert isinstance(change, p.HardIntervention)
    np.testing.assert_allclose(
        pair.after.scores.numpy(), np.full((16, 1), 2 * change.values[0] - change.values[1])
    )


def test_pair_persistence_and_failed_parameter_realization(tmp_path: Path):
    import copy
    from dataclasses import replace

    import pytest

    from r0.modules.generation import load_causal_pair, save_causal_pair

    req = p.InterventionPolicyRequest(graph(), (2,))
    sampled = p.sample_intervention_policy(
        req,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    pair = p.generate_policy_pair(sampled, seed=19, task_id=23)
    save_causal_pair(tmp_path / "pair", pair, max_bytes=8 * 1024**2)
    reopened = load_causal_pair(tmp_path / "pair")
    assert reopened.after.recipe_json == pair.after.recipe_json
    with pytest.raises(ValueError, match="semantic"):
        p.compare_causal_pair(pair.before, replace(pair.after, episode_id="forged"))
    streams = [np.random.default_rng(i) for i in (3, 1, 7)]
    states = [copy.deepcopy(r.bit_generator.state) for r in streams]
    tree = p.FrozenRFFFunction(((1.0,),), (1.0,), ((1.0,),))
    base = graph()
    base = replace(
        base, nodes=(base.nodes[0], replace(base.nodes[1], function=tree), base.nodes[2])
    )
    req = p.InterventionPolicyRequest(
        base,
        (2,),
        kind="soft",
        profile="explicit-policy-r0-v1",
        eligibility=(1,),
        parameter_laws=(p.InterventionParameterLaw(1, ("phases",), "choice-r0", choices=(-1.0,)),),
    )
    with pytest.raises(ValueError, match="phase|Fourier"):
        p.sample_intervention_policy(
            req, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
        )
    assert states == [r.bit_generator.state for r in streams]


def test_raw_decoder_admission_before_snapshot_restore():
    import copy
    from unittest.mock import patch

    import pytest

    req = p.InterventionPolicyRequest(graph(), (2,))
    sampled = p.sample_intervention_policy(
        req,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    for field in ("numbers", "working_bytes", "draws"):
        data = copy.deepcopy(asdict(sampled))
        data["request"]["raw_limits"][field] = 1
        with (
            patch.object(
                type(sampled.initial[0]),
                "restore",
                side_effect=AssertionError("snapshot restored before raw admission"),
            ),
            pytest.raises(ValueError, match="raw draw|working"),
        ):
            p.recipe_from_dict(data)


def test_classification_policy_keeps_frozen_adapter_and_schema():
    import json
    from dataclasses import replace

    import pytest

    for kind, classes in (("binary", 2), ("multiclass", 3)):
        base = graph()
        outputs = classes if kind == "multiclass" else 1
        end = replace(
            base.nodes[2],
            function=p.FrozenLinearFunction(tuple((float(i + 1),) for i in range(outputs))),
        )
        base = replace(
            base,
            task=replace(base.task, task_type=kind, classes=classes),
            nodes=(*base.nodes[:2], end),
            target=tuple((2, i) for i in range(outputs)),
        )
        sampled = p.sample_intervention_policy(
            p.InterventionPolicyRequest(base, (2,), profile="explicit-policy-r0-v1"),
            selection_rng=np.random.default_rng(3),
            value_rng=np.random.default_rng(5),
            calibration_rng=np.random.default_rng(7),
        )
        pair = p.generate_policy_pair(sampled, seed=11, task_id=13)
        assert pair.after.classes == pair.before.classes
        assert pair.after.probabilities is not None
        np.testing.assert_allclose(pair.after.probabilities.sum(1).numpy(), np.ones(16))
        trace = p.explain_generation(pair.after, [7, 0, 7])
        assert trace["episode_id"] == pair.after.episode_id
        record = json.loads(pair.after.recipe_json)
        record["execution"]["implementation"] = "forged"
        with pytest.raises(ValueError, match="execution"):
            p.explain_generation(replace(pair.after, recipe_json=json.dumps(record)), [0])


def test_parameter_overflow_and_alias_paths_never_publish_streams():
    import copy

    import pytest

    for path in (("matrix", "00"), ("matrix", "-1")):
        with pytest.raises(ValueError, match="parameter path"):
            p.InterventionPolicyRequest(
                graph(),
                (2,),
                kind="soft",
                profile="explicit-policy-r0-v1",
                eligibility=(1,),
                parameter_laws=(p.InterventionParameterLaw(1, path),),
            )
    req = p.InterventionPolicyRequest(
        graph(),
        (2,),
        kind="soft",
        profile="explicit-policy-r0-v1",
        eligibility=(1,),
        parameter_laws=(p.InterventionParameterLaw(1, ("matrix",), location=1e308, scale=1e308),),
    )
    streams = [np.random.default_rng(i) for i in (3, 13, 7)]
    before = [copy.deepcopy(r.bit_generator.state) for r in streams]
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="nonfinite"):
        p.sample_intervention_policy(
            req, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
        )
    assert [r.bit_generator.state for r in streams] == before


def test_independent_bounds_are_query_invariant_and_source_bounds_reconstruct():
    from dataclasses import replace

    for lifecycle in ("independent-calibration-r0", "source-reference-population"):
        request = p.InterventionPolicyRequest(
            graph(), (2,), reference_rows=16, bounds_lifecycle=lifecycle
        )
        sampled = p.sample_intervention_policy(
            request,
            selection_rng=np.random.default_rng(3),
            value_rng=np.random.default_rng(5),
            calibration_rng=np.random.default_rng(7),
        )
        changed = sampled.with_task(replace(sampled.task, rows=24, context_rows=9))
        assert changed.selected == sampled.selected
        if lifecycle == "independent-calibration-r0":
            assert changed.bounds == sampled.bounds
            assert changed.reference_raw == sampled.reference_raw
            assert changed.runtime.interventions == sampled.runtime.interventions
        else:
            assert changed.request.reference_rows == 24
            assert changed.reference_raw != sampled.reference_raw
        assert p.recipe_from_dict(asdict(changed)) == changed


def test_policy_outer_limits_admit_without_rewriting_retained_base():
    import pytest

    request = p.InterventionPolicyRequest(
        graph(), (2,), limits=p.CausalLimits(working_bytes=128 * 1024**2)
    )
    sampled = p.sample_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )
    for limits in (None, p.CausalLimits(working_bytes=64 * 1024**2)):
        replay = p.recipe_from_dict(asdict(sampled), limits=limits)
        assert asdict(replay) == asdict(sampled)
    with pytest.raises(ValueError, match="limit exceeded"):
        p.recipe_from_dict(asdict(sampled), limits=p.CausalLimits(working_bytes=1))
