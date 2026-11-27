"""Known policy reservations reject before replaying any retained base stream."""

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


def receipt() -> p.SampledTypedInterventionPolicy:
    cap = p.CausalLimits(working_bytes=32 * 1024**2)
    domain = p.NativeDomainPrior(kinds=("continuous",))
    functions = p.JointFunctionPrior(
        functions="lin", matrix=p.JointMatrixPrior(families=("gaussian",))
    )
    source = p.NativeVectorGraphRequest(
        ((), (0,)),
        ((domain,), (domain,)),
        ((0, 0),),
        (1, 0),
        rows=16,
        context_rows=8,
        calibration_rows=8,
        root_prior=p.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
        limits=cap,
        raw_limits=p.RawDrawLimits(state_bytes=1024, working_bytes=cap.working_bytes),
    )
    base = p.sample_native_vector_graph(
        source,
        named_rng=np.random.default_rng(71),
        raw_rng=np.random.default_rng(83),
        calibration_rng=np.random.default_rng(97),
        fit_rng=np.random.default_rng(101),
    )
    request = p.TypedInterventionPolicyRequest(
        base,
        (1,),
        (p.InterventionPolicyLocation(0, "native-variable", variable=0, location="native-value"),),
        eligibility=(0,),
        count=1,
        reference_rows=16,
        raw_limits=p.RawDrawLimits(state_bytes=8 * 1024**2),
    )
    return p.sample_typed_intervention_policy(
        request,
        selection_rng=np.random.default_rng(3),
        value_rng=np.random.default_rng(5),
        calibration_rng=np.random.default_rng(7),
    )


@pytest.mark.parametrize("cap_mib", [32, 48, 49])
@pytest.mark.parametrize("binding", ["caller", "declared"])
def test_known_outer_reservations_precede_nested_snapshot_restore(
    cap_mib: int, binding: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sampled = receipt()
    cap = p.CausalLimits(working_bytes=cap_mib * 1024**2)
    # Establish that this supported retained base is valid under the same cap.
    p.recipe_from_dict(asdict(sampled.request.base), limits=cap)
    raw = asdict(sampled)
    if binding == "declared":
        raw["request"]["limits"] = asdict(cap)

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("nested snapshot restored before known outer policy admission")

    monkeypatch.setattr(type(sampled.initial[0]), "restore", forbidden)
    with pytest.raises(ValueError, match="limit|reservation"):
        p.recipe_from_dict(raw, limits=cap if binding == "caller" else p.CausalLimits())


def test_valid_receipt_retains_original_declared_caps_and_all_custody() -> None:
    sampled = receipt()
    raw = asdict(sampled)
    decoded = p.recipe_from_dict(raw)
    assert asdict(decoded) == raw
    assert isinstance(decoded, p.SampledTypedInterventionPolicy)
    assert decoded.request.limits.working_bytes == 256 * 1024**2
    assert asdict(decoded.request.base)["request"]["limits"]["working_bytes"] == 32 * 1024**2


def test_actual_admission_transition_rejects_before_any_base_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled = receipt()
    raw = asdict(sampled)
    low, high = 32 * 1024**2, 64 * 1024**2
    calls: list[int] = []
    original = type(sampled.initial[0]).restore

    def observe(self: Any, *args: Any, **kwargs: Any):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(sampled.initial[0]), "restore", observe)
    while high - low > 1:
        cap = (low + high) // 2
        calls.clear()
        try:
            decoded = p.recipe_from_dict(raw, limits=p.CausalLimits(working_bytes=cap))
        except ValueError:
            assert not calls, f"late admission at {cap} bytes"
            low = cap
        else:
            assert asdict(decoded) == raw
            high = cap
    calls.clear()
    with pytest.raises(ValueError, match="limit|admission|reservation"):
        p.recipe_from_dict(raw, limits=p.CausalLimits(working_bytes=low))
    assert not calls
    assert asdict(p.recipe_from_dict(raw, limits=p.CausalLimits(working_bytes=high))) == raw


def test_sampling_fitted_base_checks_aggregate_before_fit_custody_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skeleton = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),))),
            p.FunctionalCausalNode((1,), p.FrozenLinearFunction(((3.0,),))),
        ),
        ((0, 0),),
        ((2, 0),),
    )
    fitted = p.sample_fitted_graph(
        skeleton,
        ((1, p.FittedTreePlan(rows=32)),),
        prefix_rng=np.random.default_rng(41),
        input_rng=np.random.default_rng(43),
        target_rng=np.random.default_rng(47),
        backend_rng=np.random.default_rng(53),
    )
    request = p.TypedInterventionPolicyRequest(
        fitted,
        (2,),
        (p.InterventionPolicyLocation(1, channels=(0,)),),
        eligibility=(1,),
        count=1,
        reference_rows=16,
        limits=p.CausalLimits(working_bytes=49 * 1024**2),
        raw_limits=p.RawDrawLimits(state_bytes=8 * 1024**2),
    )
    streams = [np.random.default_rng(n) for n in (59, 61, 67)]
    before = [str(r.bit_generator.state) for r in streams]

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("fit custody replay preceded policy aggregate admission")

    monkeypatch.setattr(type(fitted.fits[0].prefix_initial), "restore", forbidden)
    with pytest.raises(ValueError, match="limit|admission|reservation"):
        p.sample_typed_intervention_policy(
            request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
        )
    assert [str(r.bit_generator.state) for r in streams] == before
