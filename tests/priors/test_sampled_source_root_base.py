"""Public independent root provenance, arithmetic, admission and graph lifecycle."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import r0.modules.priors as p


def profile():
    return p.sample_source_root_profile(
        p.SourceRootProfile(
            3, selector="forced-r0", families=("gaussian", "categorical", "zipf_clipped")
        ),
        model_rng=np.random.default_rng(20),
        selection_rng=np.random.default_rng(21),
    )


def base(engine: str = "PCG64"):
    return p.sample_source_root_base(
        profile(),
        modes=("standardize", "identity", "center"),
        calibration_rows=64,
        calibration_rng=np.random.Generator(getattr(np.random, engine)(25)),
    )


def recipe(root: p.SourceRootBase):
    return p.NetworkRecipe(
        p.LinearRecipe(rows=32, features=3, signal_features=3, active_features=3, context_rows=16),
        p.FrozenNetworkFunction(
            3, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 3.0, -1.0),), (1.0,)),)),)
        ),
        root,
    )


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_retained_calibration_matches_independent_numpy(engine: str):
    root = base(engine)
    rng = np.random.Generator(getattr(np.random, engine)(25))
    columns: list[NDArray[Any]] = []
    for coordinate in root.law.coordinates:
        if coordinate.family == "gaussian":
            values = rng.standard_normal(64)
        elif coordinate.family == "categorical":
            values = rng.choice(len(coordinate.probabilities), size=64, p=coordinate.probabilities)
        else:
            assert coordinate.clip is not None
            values = np.minimum(rng.zipf(coordinate.exponent, 64), coordinate.clip)
        columns.append(values)
    fit = root.calibration
    assert fit is not None
    assert fit.locations == (np.mean(columns[0]), 0.0, np.mean(columns[2]))
    assert fit.scales == (np.std(columns[0], ddof=1), 1.0, 1.0)
    assert root.calibration_final is not None
    np.testing.assert_equal(
        root.calibration_final.restore().bit_generator.state, rng.bit_generator.state
    )
    assert p.sampled_source_root_base_from_dict(json.loads(json.dumps(asdict(root)))) == root
    # Apply frozen statistics to separate rows; never recompute episode moments.
    query = np.array([[2.0, 1.0, 7.0], [5.0, 0.0, 3.0]], dtype=np.float64)
    np.testing.assert_array_equal(fit.apply(query)[0], fit.apply(query[:1])[0])


def test_network_graph_archive_and_inspection_preserve_source_envelope():
    source = recipe(base())
    assert p.recipe_from_dict(asdict(source)) == source
    assert p.recipe_from_dict(asdict(source.runtime)) == source.runtime
    task = p.generate_task(source, seed=39, task_id=2**54 + 3)
    torch.testing.assert_close(
        task.scores[:, 0], task.inputs @ torch.tensor([2.0, 3.0, -1.0], dtype=task.inputs.dtype) + 1
    )
    trace = p.explain_generation(task, [31, 2, 31])
    assert trace["signal_max_error"] == 0
    assert isinstance(source.root, p.SampledSourceRootBase)
    assert json.loads(task.recipe_json)["recipe"]["root"]["profile"] == json.loads(
        json.dumps(asdict(source.root.profile))
    )


def test_identity_retains_selection_but_no_calibration():
    root = p.sample_source_root_base(profile(), modes=("identity",) * 3)
    assert root.calibration is None
    assert root.calibration_raw == ()
    assert root.calibration_initial is None
    assert root.calibration_final is None
    assert p.sampled_source_root_base_from_dict(asdict(root)) == root
    with pytest.raises(ValueError, match="root|calibration|snapshot|archive|limit|normalization"):
        p.sample_source_root_base(
            profile(), modes=("identity",) * 3, calibration_rng=np.random.default_rng(1)
        )


@pytest.mark.parametrize(
    "path",
    [
        ("law", "coordinates", 0, "location"),
        ("calibration", "locations", 0),
        ("calibration_raw", 0, "values", 0),
        ("profile", "raw", 0, "values", 0),
        ("calibration_initial", "digest"),
        ("calibration_final", "digest"),
        ("profile", "final_rng", 0, "digest"),
        ("calibration_rows",),
        ("version",),
    ],
)
def test_tampered_provenance_cannot_reopen(path: tuple[str | int, ...]):
    data = json.loads(json.dumps(asdict(base())))
    cursor = data
    for key in path[:-1]:
        cursor = cursor[key]
    value = cursor[path[-1]]
    cursor[path[-1]] = value + 1 if isinstance(value, (int, float)) else "changed"
    with pytest.raises(ValueError, match="root|calibration|snapshot|archive|limit|normalization"):
        p.sampled_source_root_base_from_dict(data)


@pytest.mark.parametrize(
    "controls",
    [
        {"limits": p.CausalLimits(working_bytes=1024)},
        {"limits": p.CausalLimits(parameters=5)},
        {"raw_limits": p.RawDrawLimits(draws=2)},
        {"raw_limits": p.RawDrawLimits(numbers=191)},
        {"raw_limits": p.RawDrawLimits(working_bytes=1024)},
    ],
)
def test_aggregate_caps_before_rng_transaction_or_snapshot(
    monkeypatch: pytest.MonkeyPatch, controls: dict[str, Any]
):
    selected = profile()
    calls: list[bool] = []

    def forbidden(*args: Any, **kwargs: Any):
        calls.append(True)
        raise AssertionError("RNG transaction/snapshot occurred before known rejection")

    monkeypatch.setattr(p.RawDrawLedger, "transaction", forbidden)
    monkeypatch.setattr(type(selected.initial_rng[0]), "restore", forbidden)
    with pytest.raises(ValueError, match="root|calibration|snapshot|archive|limit|normalization"):
        p.sample_source_root_base(
            selected,
            modes=("standardize", "identity", "center"),
            calibration_rows=64,
            calibration_rng=np.random.default_rng(25),
            **controls,
        )
    assert calls == []


def test_decode_aggregate_cap_precedes_profile_replay(monkeypatch: pytest.MonkeyPatch):
    root = base()
    data = asdict(root)

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("snapshot restored before aggregate fit admission")

    monkeypatch.setattr(type(root.calibration_initial), "restore", forbidden)
    with pytest.raises(ValueError, match="root|calibration|snapshot|archive|limit|normalization"):
        p.sampled_source_root_base_from_dict(
            data, limits=p.CausalLimits(working_bytes=10 * 1024**2)
        )


def test_no_epsilon_or_retry_and_caller_state_rollback():
    selected = p.sample_source_root_profile(
        p.SourceRootProfile(1, selector="forced-r0", families=("categorical",)),
        model_rng=np.random.default_rng(1),
        selection_rng=np.random.default_rng(2),
    )
    # Find an exact constant two-row calibration without advancing the caller stream.
    seed = next(
        i
        for i in range(100)
        if len(
            set(
                np.random.default_rng(i).choice(
                    len(selected.law.coordinates[0].probabilities),
                    2,
                    p=selected.law.coordinates[0].probabilities,
                )
            )
        )
        == 1
    )
    rng = np.random.default_rng(seed)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="degenerate"):
        p.sample_source_root_base(
            selected, modes=("standardize",), calibration_rows=2, calibration_rng=rng
        )
    assert rng.bit_generator.state == before


def test_generation_rechecks_profile_custody(monkeypatch: pytest.MonkeyPatch):
    source = recipe(base())

    def reject(*args: Any, **kwargs: Any):
        raise ValueError("source selection replay sentinel")

    monkeypatch.setattr(p.SampledSourceRootProfile, "validate_replay", reject)
    with pytest.raises(ValueError, match="source selection replay sentinel"):
        p.generate_task(source, seed=3, task_id=9)


def test_authored_base_version_and_record_remain_distinct():
    root = base()
    authored = p.SourceRootBase(root.law, root.calibration)
    assert set(asdict(authored)) == {"law", "calibration", "version"}
    assert authored.version == "source-root-base-v1"
    assert p.source_root_base_from_dict(asdict(authored)) == authored
    with pytest.raises(ValueError, match="root|calibration|snapshot|archive|limit|normalization"):
        p.source_root_base_from_dict(asdict(root))
    with pytest.raises(ValueError, match="root|calibration|snapshot|archive|limit|normalization"):
        replace(root, modes=("identity",) * 3)


@pytest.mark.parametrize(
    "family",
    [
        "gaussian",
        "presampled_gaussian",
        "unit_uniform",
        "symmetric_uniform",
        "categorical",
        "zipf_clipped",
        "zipf_unbounded",
    ],
)
def test_all_selected_families_retain_law_and_raw_profile(family: str):
    selected = p.sample_source_root_profile(
        p.SourceRootProfile(1, selector="forced-r0", families=(family,)),
        model_rng=np.random.default_rng(7),
        selection_rng=np.random.default_rng(8),
    )
    root = p.sample_source_root_base(selected, modes=("identity",))
    assert root.profile == selected
    assert root.law == selected.law
    assert root.moment_order == selected.law.moment_order
    assert p.sampled_source_root_base_from_dict(asdict(root)) == root


def test_sampled_root_matching_roundtrip_and_trace():
    source = recipe(base())
    matching = p.fit_network_variance_match(
        source,
        before_inputs=(0, 1, 2),
        after_inputs=(1, 2),
        rows=47,
        rng=np.random.default_rng(50),
    )
    view = p.MatchedNetworkInputView(
        source, replace(source.task, active_features=2), (1, 2), matching=matching
    )
    assert p.recipe_from_dict(asdict(view)) == view
    task = p.generate_task(view, seed=8, task_id=2**54 + 11)
    assert p.explain_generation(task, [31, 0, 31])["signal_max_error"] == 0


def test_graph_budget_accounts_for_root_replay_before_generation():
    source = recipe(base())
    # This covers source profile replay but not the aggregate independent fit.
    with pytest.raises(ValueError, match="sampled root aggregate working"):
        replace(source, limits=p.CausalLimits(working_bytes=10 * 1024**2))


def test_profile_and_calibration_custody_are_both_required():
    root = base()
    assert root.calibration is not None
    changed = asdict(root)
    changed["profile"] = asdict(
        p.sample_source_root_profile(
            root.profile.profile,
            model_rng=np.random.default_rng(20),
            selection_rng=np.random.default_rng(999),
        )
    )
    with pytest.raises(ValueError, match="law differs"):
        p.sampled_source_root_base_from_dict(changed)
    changed = asdict(root)
    changed["calibration"]["rows"] += 1
    with pytest.raises(ValueError, match="archive differs"):
        p.sampled_source_root_base_from_dict(changed)


def test_persisted_task_retains_exact_root_recipe(tmp_path: Path):
    from r0.modules import generation as g

    source = recipe(base())
    identity = 2**54 + 3
    task = p.generate_task(source, seed=39, task_id=identity)
    path = tmp_path / "sampled-root.zip"
    g.save_group(path, [(identity, task)], max_bytes=4 * 1024**2)
    reopened = next(g.load_group(path))
    assert reopened.recipe_json == task.recipe_json
    decoded = p.recipe_from_dict(json.loads(reopened.recipe_json)["recipe"])
    replayed = p.generate_task(decoded, seed=39, task_id=identity)
    assert torch.equal(replayed.inputs, reopened.inputs)
    assert torch.equal(replayed.targets, reopened.targets)


@pytest.mark.parametrize("kind", ["native", "vector", "converted"])
def test_typed_graphs_use_the_sampled_root_adapter(kind: str):
    selected = p.sample_source_root_profile(
        p.SourceRootProfile(1, selector="p1-code-normal"),
        model_rng=np.random.default_rng(1),
        selection_rng=np.random.default_rng(2),
    )
    root = p.sample_source_root_base(
        selected, modes=("center",), calibration_rows=8, calibration_rng=np.random.default_rng(3)
    )
    nodes = (
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,),)), root=root, version="functional-causal-node-v2"
        ),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0,),))),
    )
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    source: Any
    if kind == "native":
        source = p.NativeCausalRecipe(
            task, tuple(p.NativeCausalNode(n) for n in nodes), ((0, 0),), (1, 0)
        )
    elif kind == "vector":
        source = p.NativeVectorRecipe(
            task,
            tuple(p.NativeVectorNode(n, (p.VariableDomain(),)) for n in nodes),
            ((0, 0),),
            (1, 0),
        )
    else:
        source = p.ConvertedCausalRecipe(
            task,
            tuple(p.ConvertedCausalNode(n) for n in nodes),
            (p.ConvertedReadout(0, channel=0),),
            p.ConvertedReadout(1, channel=0),
        )
    assert p.recipe_from_dict(asdict(source)) == source
    emitted = p.generate_task(source, seed=19, task_id=77)
    torch.testing.assert_close(emitted.targets, 2 * emitted.inputs[:, 0])
    trace = p.explain_generation(emitted, [15, 0, 15])
    assert trace["rows"] == [15, 0, 15]
    values = next(node["values"] for node in trace["nodes"] if node["id"] == "function-1")
    np.testing.assert_allclose(np.asarray(values)[:, 0], emitted.targets[[15, 0, 15]].numpy())


@pytest.mark.parametrize(
    "field", ["profile", "law", "calibration_rows", "limits", "raw_limits", "modes"]
)
def test_incomplete_root_archives_raise_value_error(field: str):
    data = asdict(base())
    del data[field]
    with pytest.raises(ValueError, match="malformed|fields"):
        p.sampled_source_root_base_from_dict(data)


def test_finite_calibration_does_not_upgrade_zipf_population_moments():
    selected = p.sample_source_root_profile(
        p.SourceRootProfile(1, selector="forced-r0", families=("zipf_unbounded",)),
        model_rng=np.random.default_rng(1),
        selection_rng=np.random.default_rng(2),
    )
    root = p.sample_source_root_base(
        selected,
        modes=("standardize",),
        calibration_rows=64,
        calibration_rng=np.random.default_rng(3),
    )
    assert root.moment_order == 1.5
    nodes = (
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,),)), root, version="functional-causal-node-v2"
        ),
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
    )
    with pytest.raises(ValueError, match="finite second moments"):
        p.CalibratedFunctionalCausalRecipe(
            p.LinearRecipe(
                rows=32, context_rows=16, features=1, signal_features=1, active_features=1
            ),
            nodes,
            ((0, 0),),
            ((1, 0),),
        )
