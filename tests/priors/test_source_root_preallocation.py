"""Known aggregate raw failures precede output allocation and all stochastic work."""

import copy
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


def deny_output_allocation(shape: Any, *args: Any, **kwargs: Any) -> Any:
    if type(shape) is tuple and len(cast(tuple[Any, ...], shape)) == 2:
        raise AssertionError("output allocated before admission")
    return original_empty(cast(Any, shape), *args, **kwargs)


original_empty = np.empty


def root():
    return p.SourceRootLaw((p.SourceRootCoordinate("unit_uniform"), p.SourceRootCoordinate("zipf")))


@pytest.mark.parametrize("cap", ["draws", "numbers", "working_bytes"])
@pytest.mark.parametrize("existing", [False, True])
def test_all_raw_caps_reject_before_array_allocation(cap: str, existing: bool):
    # Independent cost arithmetic: one uniform record and one Zipf record, each 13 scalars.
    owner = ("source-root",)
    costs = [
        96 * 13
        + 128 * (metadata + 1 + 2)
        + 1024
        + len("source-rootcolumn:0valuesraw")
        + len(distribution)
        for metadata, distribution in [(0, "random"), (1, "zipf")]
    ]
    consumed = 1 if existing else 0
    prior_cost = (
        (96 + 128 * 3 + 1024 + len("source-rootcolumn:0valuesrawrandom")) if existing else 0
    )
    limits = p.RawDrawLimits(
        draws=consumed + 2 - (cap == "draws"),
        numbers=consumed + 26 - (cap == "numbers"),
        working_bytes=2 * 1024**2 + sum(costs) + prior_cost - (cap == "working_bytes"),
    )
    rng = np.random.default_rng(719)
    ledger = p.RawDrawLedger(limits=limits)
    if existing:
        p.draw_source_root(
            p.SourceRootLaw((p.SourceRootCoordinate("unit_uniform"),)),
            rows=1,
            rng=rng,
            capture=ledger,
        )
    before, records = copy.deepcopy(rng), ledger.freeze()
    with (
        patch.object(np, "empty", side_effect=deny_output_allocation),
        pytest.raises(ValueError, match="retained record limit"),
    ):
        p.draw_source_root(root(), rows=13, rng=rng, capture=ledger, owner=owner)
    assert ledger.freeze() == records
    np.testing.assert_array_equal(rng.random(9), before.random(9))


@pytest.mark.parametrize("owner", ["text", (), ("bad ",), ("x",) * 16])
def test_invalid_root_owner_rejects_before_array_allocation(owner: Any):
    rng = np.random.default_rng(7)
    with (
        patch.object(np, "empty", side_effect=deny_output_allocation),
        pytest.raises(ValueError, match="joint"),
    ):
        p.draw_source_root(root(), rows=2, rng=rng, owner=owner)


def test_exact_aggregate_cost_and_nested_stream_admission():
    # Long non-default stream metadata must use its actual encoded length.
    name = "a-longer-owned-stream"
    request = p.RawDrawSpec(
        ("root", "channel"),
        "values",
        "choice",
        (7,),
        (("count", 3), ("probabilities", (0.2, 0.3, 0.5))),
    )
    cost = 96 * 7 + 128 * (4 + 1 + 2) + 1024 + len("rootchannelvalueschoice" + name)
    assert request.retained_bytes(name) == cost
    rng = np.random.default_rng(18)
    ledger = p.RawDrawLedger(
        limits=p.RawDrawLimits(draws=1, numbers=7, working_bytes=cost + 2 * 1024**2)
    )
    with ledger.transactions({name: rng}) as (streams, staged):
        before = copy.deepcopy(streams[name])
        staged.admit_draws(streams[name], (request,))
        np.testing.assert_equal(streams[name].bit_generator.state, before.bit_generator.state)
        staged.draw(
            streams[name],
            owner=request.owner,
            role=request.role,
            distribution=request.distribution,
            shape=request.shape,
            parameters=request.parameters,
        )
        with pytest.raises(ValueError, match="retained record limit"):
            staged.admit_draws(streams[name], (request,))
    assert len(ledger.freeze()) == 1
