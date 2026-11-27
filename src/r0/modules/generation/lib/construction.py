"""Graph construction runs inside the guarded generation worker."""

import json
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np

from r0.modules import priors
from r0.modules.generation.lib.fitted_construction import construct_fitted
from r0.modules.generation.lib.native_construction import construct_native
from r0.modules.generation.lib.network_construction import construct_network
from r0.modules.generation.lib.sampled_fitted_construction import construct_sampled_fitted


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
    if topology in ("network", "network-authored"):
        return construct_network(
            fields, seed=seed, limits=limits, authored=topology == "network-authored"
        )
    if topology == "sampled-fitted":
        return construct_sampled_fitted(fields, seed=seed, limits=limits)
    if topology == "fitted":
        return construct_fitted(fields, seed=seed, limits=limits)
    if topology == "sampled-native-vector":
        request = priors.native_vector_request_from_dict(
            {**fields, "limits": asdict(limits)}, limits=limits
        )
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
        return priors.sample_native_vector_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
    if topology == "sampled-native":
        return construct_native(fields, seed=seed, limits=limits)
    task = priors.recipe_from_dict(fields.pop("task"))
    if type(task) is not priors.LinearRecipe:
        raise ValueError("graph construction requires linear task controls")
    if topology in ("native", "native-vector"):
        native = priors.recipe_from_dict(
            {"task": asdict(task), "limits": asdict(limits), **fields}, limits=limits
        )
        if type(native) not in (priors.NativeCausalRecipe, priors.NativeVectorRecipe):
            raise ValueError("native construction requires an authored native recipe")
        return native
    raw_composition = fields.pop("composition", None)
    composition = (
        None if raw_composition is None else priors.ParentCompositionPrior(**raw_composition)
    )
    if composition is not None and topology == "cauchy":
        raise ValueError("parent composition uses explicit or geometry topology controls")
    functions = fields.pop("function_prior")
    matrix = priors.JointMatrixPrior(**functions.pop("matrix"))
    function = priors.JointFunctionPrior(matrix=matrix, **functions)
    roots = priors.JointRootPrior(families=fields.pop("root_families"), function=function)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    rngs = dict(
        named_rng=streams[0], raw_rng=streams[1], calibration_rng=streams[2], fit_rng=streams[3]
    )
    if topology in ("geometry", "attachment-geometry", "motif-geometry"):
        geometry_prior: (
            priors.CauchyGraphPrior | priors.AttachmentGraphPrior | priors.MotifGraphPrior
        )
        if topology == "geometry":
            geometry_prior = priors.CauchyGraphPrior(
                nodes=fields.pop("nodes"), offset=fields.pop("offset")
            )
        elif topology == "attachment-geometry":
            geometry_prior = priors.AttachmentGraphPrior(**fields.pop("topology_prior"))
        else:
            geometry_prior = priors.MotifGraphPrior(**fields.pop("topology_prior"))
        widths = priors.JointWidthPrior(**fields.pop("width_prior"))
        pipeline = fields.pop("pipeline_prior", None)
        request_type = priors.GeometryGraphRequest
        if pipeline is not None:
            fields["pipeline_prior"] = priors.pipeline_prior_from_dict(pipeline)
            request_type = priors.GeometryPipelineRequest
        if composition is not None:
            request_type = priors.GeometryParentRequest
            fields["composition"] = composition
        geometry = request_type(
            task=task,
            topology=geometry_prior,
            width_prior=widths,
            root_prior=roots,
            function_prior=function,
            limits=limits,
            **fields,
        )
        return priors.sample_geometry_graph(geometry, **rngs)
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
        if composition is not None:
            return priors.sample_parent_graph(request, composition=composition, **rngs)
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

    def task_record(self) -> dict[str, Any]:
        """Return preconstruction row geometry; sampled native target type is not yet known."""
        controls = json.loads(self.controls_json)
        if controls.get("topology") in (
            "fitted",
            "sampled-fitted",
            "population-fitted",
            "sampled-population-fitted",
        ):
            return controls["skeleton"]["task"]
        if controls.get("topology") not in ("sampled-native", "sampled-native-vector"):
            return controls["task"]
        observed = controls["observed"]
        if type(observed) is not list:
            raise ValueError("sampled native observed channels must be an array")
        count = len(cast(list[Any], observed))
        return asdict(
            priors.LinearRecipe(
                rows=controls.get("rows", 64),
                context_rows=controls.get("context_rows", 48),
                split=controls.get("split", 0),
                features=count,
                signal_features=count,
                active_features=count,
            )
        )

    def build(
        self, limits: priors.CausalLimits, *, episode_seed: int = 0, task_id: int = 0
    ) -> priors.Recipe:
        fields = json.loads(self.controls_json)
        if fields.get("topology") == "sampled-population-fitted":
            from r0.modules.generation.lib.sampled_population_construction import (
                construct_sampled_population,
            )

            return construct_sampled_population(
                fields,
                construction_seed=self.seed,
                seed=episode_seed,
                task_id=task_id,
                limits=limits,
            )
        if fields.get("topology") == "population-fitted":
            from r0.modules.generation.lib.population_fitted_construction import (
                construct_population_fitted,
            )

            return construct_population_fitted(
                fields,
                construction_seed=self.seed,
                seed=episode_seed,
                task_id=task_id,
                limits=limits,
            )
        return construct_graph(fields, seed=self.seed, limits=limits)
