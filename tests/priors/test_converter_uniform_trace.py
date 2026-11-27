"""A retained stochastic representation keeps its original emission-uniform edge."""

import pytest
import torch

from r0.modules import priors as p


@pytest.mark.parametrize(
    ("mode", "location", "retains_uniform"),
    [
        ("softmax_disc", "capture", True),
        ("softmax_int", "capture", True),
        ("softmax_id", "capture", False),
        ("softmax_disc", "representation_and_capture", False),
        ("softmax_int", "representation_and_capture", False),
    ],
)
def test_converter_representation_traces_only_executed_emission_uniforms(
    mode: str, location: str, retains_uniform: bool
) -> None:
    converter = p.CategoricalConverter(
        "choice",
        0,
        2,
        ("A", "B"),
        mode,
        ((-2.0, -3.0), (4.0, 5.0)),
        mean=(0.0, 0.0),
        scale=(1.0, 1.0),
        bias=(0.0, 0.0),
    )
    graph = p.ConvertedCausalRecipe(
        p.LinearRecipe(rows=32, context_rows=16, features=1, signal_features=1, active_features=1),
        (
            p.ConvertedCausalNode(
                p.FunctionalCausalNode(
                    (),
                    p.FrozenLinearFunction(((0.0,), (0.0,))),
                    p.JointRootBase("gaussian", 1),
                ),
                (converter,),
            ),
            p.ConvertedCausalNode(
                p.FunctionalCausalNode(
                    (0,),
                    p.FrozenLinearFunction(((1.0, 0.0),)),
                )
            ),
        ),
        (p.ConvertedReadout(0, capture="choice"),),
        p.ConvertedReadout(1, channel=0),
    )
    pair = p.generate_causal_pair(
        graph,
        interventions=(p.ConverterHardIntervention(0, "choice", "A", location),),
        seed=179,
        task_id=23,
    )
    assert torch.all(pair.after.inputs == 0)
    if location == "capture":
        assert torch.equal(pair.before.targets, pair.after.targets)
    if retains_uniform:
        assert set(pair.after.targets.numpy().tolist()) == (
            {-2.0, 4.0} if mode == "softmax_disc" else {0.0, 1.0}
        )
    trace = p.explain_generation(pair.after, [15, 0, 15])
    nodes = {n["id"]: n for n in trace["nodes"]}
    parents = nodes["intervention-0-representation:choice"]["parents"]
    uniform = "intervention-0-uniform:choice"
    assert (uniform in nodes) == retains_uniform
    assert (uniform in parents) == retains_uniform
    if location == "capture":
        assert "intervention-0-capture:choice" not in parents
        assert "intervention-0-probabilities:choice" not in parents
    else:
        assert parents == ["intervention-0-capture:choice"]
    seen: set[str] = set()
    for node in trace["nodes"]:
        assert set(node["parents"]).issubset(seen)
        seen.add(node["id"])
