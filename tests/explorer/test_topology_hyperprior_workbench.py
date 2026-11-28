"""Topology construction retains complete source requests through the guarded explorer."""

import json
from pathlib import Path

import pytest

from r0.modules.explorer import ExplorerSession


def test_joint_topology_workbench_retains_five_streams(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Topology hyperprior: joint linear"
    ]
    result = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": 28,
            "seed": 19,
            "task_id": 7,
            "causal_limits": controls["request"]["limits"],
            "max_seconds": 120,
        },
    )
    archive = json.loads(result["recipe_json"])["recipe"]
    assert archive["version"] == "sampled-joint-hyperprior-graph-v1"
    assert archive["request"] == json.loads(json.dumps(controls["request"]))
    assert len(archive["graph"]["initial"]) == 4
    assert archive["topology"]["initial"]["engine"] == "PCG64"
    assert session.request(
        "graph", {"indices": [7, 0, 7], "causal_limits": controls["request"]["limits"]}
    )["graph"]["rows"] == [7, 0, 7]


def test_authored_topology_workbench_keeps_templates(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Topology hyperprior: authored linear"
    ]
    result = session.request(
        "construct",
        {
            "controls": controls,
            "construction_seed": 28,
            "seed": 19,
            "task_id": 7,
            "causal_limits": controls["request"]["limits"],
            "max_seconds": 120,
        },
    )
    archive = json.loads(result["recipe_json"])["recipe"]
    assert archive["request"] == json.loads(json.dumps(controls["request"]))
    assert archive["request"]["edge_template"]["function"]["matrix"] == [[2.0]]
    assert session.request(
        "graph", {"indices": [7, 0, 7], "causal_limits": controls["request"]["limits"]}
    )["graph"]["rows"] == [7, 0, 7]


def test_full_source_catalogs_remain_visible_without_sampling(tmp_path: Path):
    examples = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    for token in (
        "lin",
        "quad",
        "gp",
        "tree",
        "disc",
        "em",
        "mlp",
        "prod",
        "kumo-prod",
        "default",
        "default,kumo-prod",
    ):
        raw = examples[f"Topology hyperprior: joint catalog {token}"]["request"]
        assert raw["function_prior"]["functions"] == token
        assert len(raw["function_prior"]["matrix"]["families"]) == 5
        assert len(raw["root_prior"]["families"]) == 4
    assert (
        examples["Topology hyperprior: joint log-size"]["request"]["topology"][
            "probability_transform"
        ]
        == "gamma-exp-r0"
    )


def test_invalid_request_rejects_before_stream_initialization(tmp_path: Path):
    from unittest.mock import patch

    import pytest

    from r0.modules import priors as p
    from r0.modules.generation import GraphConstruction

    examples = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    for label in ("joint linear", "authored linear"):
        controls = examples[f"Topology hyperprior: {label}"]
        controls["request"]["layout"] = "unknown"
        construction = GraphConstruction(json.dumps(controls), 28)
        with (
            patch(
                "numpy.random.SeedSequence", side_effect=AssertionError("draw before validation")
            ),
            patch("numpy.random.Generator", side_effect=AssertionError("draw before validation")),
            pytest.raises(ValueError, match="request|hyperprior|topology"),
        ):
            construction.build(p.CausalLimits(**controls["request"]["limits"]))


def test_complete_catalog_requests_admit_without_narrowing(tmp_path: Path):
    from r0.modules import priors as p

    examples = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    for label, controls in examples.items():
        if not label.startswith("Topology hyperprior: joint catalog"):
            continue
        raw = controls["request"]
        request = p.joint_hyperprior_request_from_dict(raw, limits=p.CausalLimits(**raw["limits"]))
        assert request.function_prior.functions == raw["function_prior"]["functions"]
        assert len(request.function_prior.matrix.families) == 5
        assert len(request.root_prior.families) == 4


@pytest.mark.parametrize("field", ["header", "choice", "raw_shape", "cardinality"])
def test_authored_sampled_template_headers_reject_before_restore(tmp_path: Path, field: str):
    from dataclasses import asdict
    from unittest.mock import patch

    import numpy as np
    import pytest

    from r0.modules import priors as p

    raw = ExplorerSession(tmp_path).request("construction_examples", {})["examples"][
        "Topology hyperprior: authored linear"
    ]["request"]
    sample = p.sample_feature_conv1d(
        p.FeatureConv1DRequest(1, 1, (p.Conv1DLayerPrior((p.Conv1DGeometry(1, 1),)),)),
        named_rng=np.random.default_rng(3),
        raw_rng=np.random.default_rng(4),
    )
    raw["edge_template"]["function"] = asdict(sample)
    limits = p.CausalLimits(**raw["limits"])
    valid = p.hyperprior_graph_request_from_dict(raw, limits=limits)
    assert asdict(valid)["edge_template"]["function"] == asdict(sample)
    record = raw["edge_template"]["function"]
    if field == "header":
        record["final"][1]["engine"] = "unknown"
    elif field == "choice":
        record["choices"] = ((999, 0, 0),)
    elif field == "raw_shape":
        record["raw"][0]["shape"] = (999,)
    else:
        record["initial"] = ()
    with (
        patch.object(
            type(sample.initial[0]), "restore", side_effect=AssertionError("early restore")
        ),
        pytest.raises(ValueError, match="snapshot|generator|convolution|draw|custody"),
    ):
        p.hyperprior_graph_request_from_dict(raw, limits=p.CausalLimits(**raw["limits"]))
