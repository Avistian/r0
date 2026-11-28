"""Request-derived convolution layout is checked before any retained stream replay."""

from dataclasses import asdict
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


def request_record(
    layout: str = "channel-major", placement: str = "after", scope: str = "flat"
) -> tuple[dict[str, Any], type]:
    sample = p.sample_feature_conv1d(
        p.FeatureConv1DRequest(
            1,
            1,
            (
                p.Conv1DLayerPrior(
                    (p.Conv1DGeometry(1, 1),), placement=placement, activation_scope=scope
                ),
            )
            * 2,
            layout=layout,
        ),
        named_rng=np.random.default_rng(3),
        raw_rng=np.random.default_rng(4),
    )
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    root = p.FunctionalCausalNode((), sample, p.JointRootBase("gaussian", 1))
    edge = p.FunctionalCausalNode((0,), sample)
    request = p.HyperpriorGraphRequest(task, p.FixedSizeTopologyHyperprior(3), root, edge)
    return asdict(request), type(sample.initial[0])


@pytest.mark.parametrize("template", ["root_template", "edge_template"])
@pytest.mark.parametrize("field", ["layout", "placement", "activation_scope"])
@pytest.mark.parametrize("side", ["request", "function"])
def test_template_deterministic_fields_precede_restore(template: str, field: str, side: str):
    raw, snapshot = request_record()
    record = raw[template]["function"][side]
    if field == "layout":
        record[field] = "position-major"
    else:
        record["layers"][-1][field] = "before" if field == "placement" else "channels"
    with (
        patch.object(snapshot, "restore", side_effect=AssertionError("early replay")),
        pytest.raises(ValueError, match="convolution.*(layout|configuration)"),
    ):
        p.hyperprior_graph_request_from_dict(raw, limits=p.CausalLimits())


@pytest.mark.parametrize("layout", ["channel-major", "position-major"])
@pytest.mark.parametrize(
    ("placement", "scope"), [("after", "flat"), ("before", "channels"), ("after", "positions")]
)
def test_valid_sampled_templates_keep_full_records(layout: str, placement: str, scope: str):
    raw, _ = request_record(layout, placement, scope)
    assert asdict(p.hyperprior_graph_request_from_dict(raw, limits=p.CausalLimits())) == raw
