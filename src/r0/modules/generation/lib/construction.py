"""Graph construction runs inside the guarded generation worker."""

import json
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from r0.modules import priors


def _tuples(value: Any) -> Any:
    if type(value) is list:
        return tuple(_tuples(v) for v in cast(list[Any], value))
    if type(value) is dict:
        return {k: _tuples(v) for k, v in cast(dict[str, Any], value).items()}
    return value


def construct_graph(
    controls: dict[str, Any], *, seed: int, limits: priors.CausalLimits
) -> priors.Recipe:
    """Sample bounded construction; its archive retains actual four-stream states."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("construction seed must be an integer in [0, 2**63)")
    fields = _tuples(controls)
    topology = fields.pop("topology")
    task = priors.recipe_from_dict(fields.pop("task"))
    if type(task) is not priors.LinearRecipe:
        raise ValueError("graph construction requires linear task controls")
    functions = fields.pop("function_prior")
    matrix = priors.JointMatrixPrior(**functions.pop("matrix"))
    function = priors.JointFunctionPrior(matrix=matrix, **functions)
    roots = priors.JointRootPrior(families=fields.pop("root_families"), function=function)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    rngs = dict(
        named_rng=streams[0], raw_rng=streams[1], calibration_rng=streams[2], fit_rng=streams[3]
    )
    if topology == "cauchy":
        prior = priors.CauchyGraphPrior(nodes=len(fields["widths"]), offset=fields.pop("offset"))
        request = priors.TopologyGraphRequest(
            task=task,
            topology=prior,
            root_prior=roots,
            function_prior=function,
            limits=limits,
            **fields,
        )
        return priors.sample_topology_graph(request, **rngs)
    elif topology == "explicit":
        request = priors.JointGraphRequest(
            task=task, root_prior=roots, function_prior=function, limits=limits, **fields
        )
        return priors.sample_joint_graph(request, **rngs)
    else:
        raise ValueError("unknown graph construction topology")


@dataclass(frozen=True, slots=True)
class GraphConstruction:
    controls_json: str
    seed: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("construction seed must be an integer in [0, 2**63)")
        if type(self.controls_json) is not str or len(self.controls_json.encode()) > 1024**2:
            raise ValueError("construction controls must be JSON below 1 MiB")
        if type(json.loads(self.controls_json)) is not dict:
            raise ValueError("construction controls must be an object")

    def build(self, limits: priors.CausalLimits) -> priors.Recipe:
        return construct_graph(json.loads(self.controls_json), seed=self.seed, limits=limits)
