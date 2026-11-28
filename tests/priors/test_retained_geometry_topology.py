"""An already sampled topology must not consume a second edge stream."""

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


def test_fixed_parent_geometry_retains_edges_and_draws_no_topology_noise():
    prior = p.FixedParentTopology(((), (0,), (0, 1)))
    rng = np.random.default_rng(719)
    before = rng.bit_generator.state
    record = p.retain_parent_topology(prior, limits=p.CausalLimits())
    record.validate()
    record.replay(rng)
    assert rng.bit_generator.state == before
    assert record.parents == ((), (0,), (0, 1))
    assert record.raw == ()
    assert asdict(record.prior) == asdict(prior)


@pytest.mark.parametrize("parents", [((), (1,)), ((), (True,)), ((), (0, 0)), ((), [0])])
def test_fixed_parent_topology_rejects_ambiguous_or_non_dag_geometry(parents: Any):
    with pytest.raises(ValueError, match="ordered acyclic exact parent indices"):
        p.FixedParentTopology(parents)
