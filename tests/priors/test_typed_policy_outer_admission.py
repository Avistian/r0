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


@pytest.mark.parametrize("cap_mib", [32, 48])
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
