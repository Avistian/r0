"""Reconstruct complete joint graph semantics without restoring construction streams."""

from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_decoding import functional_recipe_from_dict
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.joint_function_decoding import joint_function_plan_from_dict
from r0.modules.priors.lib.joint_function_realization_decoding import (
    joint_function_realization_from_dict,
)
from r0.modules.priors.lib.joint_graph_admission import JointGraphAdmission
from r0.modules.priors.lib.joint_graph_plans import JointGraphNodePlan, JointGraphPlan
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_plan_ownership import JointPlanOwner
from r0.modules.priors.lib.joint_registry_decoding import joint_operation_registry_from_dict
from r0.modules.priors.lib.joint_root_decoding import joint_root_plan_from_dict
from r0.modules.priors.lib.joint_root_realization import JointRootRealization
from r0.modules.priors.lib.joint_root_realization_decoding import joint_root_realization_from_dict
from r0.modules.priors.lib.joint_snapshot_preflight import validate_joint_snapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.sampled_functional_graph import (
    GraphCalibrationReceipt,
    SampledFunctionalGraph,
)
from r0.modules.priors.lib.sampled_graph_decoding import construction_matches
from r0.modules.priors.lib.topology_hyperprior_decoding import complete_topology_record


def _array(value: Any) -> tuple[Any, ...]:
    if type(value) not in (tuple, list):
        raise ValueError("joint graph metadata requires explicit arrays")
    return tuple(value)


def _tuples(value: Any) -> Any:
    if type(value) in (tuple, list):
        return tuple(_tuples(item) for item in _array(value))
    return value


def _authored_catalog(value: Any) -> None:
    """This source catalog emits authored mechanisms, never nested sampling envelopes.

    Inspect the complete tree before dispatching any frozen-function decoder: an
    injected unrelated sampled wrapper must not execute its restore path here.
    """
    if type(value) is dict:
        mapping = cast(dict[str, Any], value)
        version = mapping.get("version")
        if type(version) is str and version.startswith("sampled-"):
            raise ValueError("joint mechanism metadata cannot contain nested sampled envelopes")
        for item in mapping.values():
            _authored_catalog(item)
    elif type(value) in (tuple, list):
        for item in _array(value):
            _authored_catalog(item)


def preflight_joint_hyperprior_graph(raw: dict[str, Any], request: JointGraphRequest) -> None:
    """Whole finite-support admission has already charged this complete retained payload."""
    limits = request.limits
    for key, value in raw.items():
        if key != "version":
            _authored_catalog(value)
    runtime = functional_recipe_from_dict(raw["runtime"], limits=limits)
    if type(runtime) is not FunctionalCausalRecipe or (
        not construction_matches(request.task, raw["runtime"]["task"])
        or runtime.observed != request.observed
        or runtime.target != request.target
        or runtime.limits != limits
    ):
        raise ValueError("joint graph runtime metadata differs from request")
    registry = joint_operation_registry_from_dict(raw["registry"], limits=limits)
    planned = complete_topology_record(raw["plan"], JointGraphPlan)
    nodes: list[JointGraphNodePlan] = []
    for item in _array(planned["nodes"]):
        node = complete_topology_record(item, JointGraphNodePlan)
        parents = _array(node["parents"])
        mechanism = (
            joint_function_plan_from_dict(
                node["mechanism"], decisions=registry.decisions, limits=limits
            )
            if parents
            else joint_root_plan_from_dict(
                node["mechanism"], decisions=registry.decisions, limits=limits
            )
        )
        nodes.append(JointGraphNodePlan(parents, node["width"], mechanism))
    plan = JointGraphPlan(
        _array(planned["operation"]),
        tuple(nodes),
        planned["calibration_rows"],
        CausalLimits(**complete_topology_record(planned["limits"], CausalLimits)),
        planned["version"],
    )
    owners = tuple(
        JointPlanOwner(_array(item["operation"]), item["kind"], _array(item["reference"]))
        for item in (complete_topology_record(x, JointPlanOwner) for x in _array(raw["owners"]))
    )
    plan.validate(registry, owners=owners)
    if not construction_matches(plan.owners(), raw["owners"]):
        raise ValueError("joint graph ownership metadata has noncanonical primitive types")
    realizations = tuple(
        joint_root_realization_from_dict(item, decisions=registry.decisions, limits=limits)
        if item.get("version") == "joint-root-realization-v1"
        else joint_function_realization_from_dict(item, decisions=registry.decisions, limits=limits)
        for item in _array(raw["realizations"])
    )
    receipts: list[GraphCalibrationReceipt] = []
    for item in _array(raw["calibrations"]):
        fields = complete_topology_record(item, GraphCalibrationReceipt)
        fields["parents"] = _array(fields["parents"])
        if type(fields["node"]) is not int or any(type(i) is not int for i in fields["parents"]):
            raise ValueError("joint graph calibration locations require exact integers")
        receipts.append(GraphCalibrationReceipt(**fields))
    draws: list[RawDrawRecord] = []
    for item in _array(raw["raw"]):
        fields = complete_topology_record(item, RawDrawRecord)
        for name in ("owner", "shape", "parameters", "values"):
            fields[name] = _tuples(fields[name])
        draws.append(RawDrawRecord(**fields))
    specs = tuple(
        RawDrawSpec(d.owner, d.role, d.distribution, d.shape, d.parameters) for d in draws
    )
    caps = request.raw_limits
    if (
        len(draws) > caps.draws
        or sum(s.numbers for s in specs) > caps.numbers
        or sum(s.retained_bytes(d.stream) for s, d in zip(specs, draws, strict=True))
        + 2 * caps.state_bytes
        > caps.working_bytes
    ):
        raise ValueError("joint graph retained raw metadata exceeds request limits")
    admission = complete_topology_record(raw["admission"], JointGraphAdmission)
    admission["raw_limits"] = RawDrawLimits(
        **complete_topology_record(admission["raw_limits"], RawDrawLimits)
    )
    states: list[tuple[GeneratorSnapshot, ...]] = []
    for key in ("initial", "final"):
        snapshots = tuple(
            GeneratorSnapshot(**complete_topology_record(s, GeneratorSnapshot))
            for s in _array(raw[key])
        )
        for snapshot in snapshots:
            validate_joint_snapshot(snapshot, byte_limit=request.raw_limits.state_bytes)
        states.append(snapshots)
    for index, stream in enumerate(("named", "raw", "calibration", "fit")):
        if states[0][index].engine != states[1][index].engine:
            raise ValueError("joint graph source engine changes across construction")
        if stream == "named":
            continue  # Named choices have their own complete registry, not raw records.
        digest = states[0][index].digest
        for draw in draws:
            if draw.stream == stream:
                if draw.generator != states[0][index].engine or draw.state_before != digest:
                    raise ValueError("joint graph raw metadata stream chain differs")
                digest = draw.state_after
        if digest != states[1][index].digest:
            raise ValueError("joint graph final raw stream differs from retained chain")
    if any(d.stream not in ("raw", "calibration", "fit") for d in draws):
        raise ValueError("joint graph raw record has an undeclared stream")
    metadata = SampledFunctionalGraph(
        request,
        plan,
        registry,
        owners,
        realizations,
        tuple(receipts),
        tuple(draws),
        states[0],
        states[1],
        JointGraphAdmission(**admission),
        runtime,
        raw["version"],
    )
    metadata.validate_structure()
    canonical_nodes: list[FunctionalCausalNode] = []
    for i, realization in enumerate(realizations):
        if isinstance(realization, JointRootRealization):
            function, root = realization.function.function, realization.calibration.base
        else:
            function, root = realization.function, None
        noise = request.noise[i] if request.noise else JointGraphNoise()
        pipeline = request.authored_pipelines[i] if request.authored_pipelines else None
        canonical_nodes.append(
            FunctionalCausalNode(
                request.parents[i],
                function,
                root,
                noise.scales,
                noise.exogenous,
                noise.response,
                pipeline,
            )
        )
    canonical_runtime = FunctionalCausalRecipe(
        request.task, tuple(canonical_nodes), request.observed, request.target, request.limits
    )
    if not construction_matches(canonical_runtime, raw["runtime"]):
        raise ValueError("joint graph runtime differs from exact owned realization fields")
    if not construction_matches(metadata, raw):
        raise ValueError("joint graph metadata must retain exact complete nested fields")
