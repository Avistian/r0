"""Complete hyperprior decoding admits worst geometry before snapshot restoration."""

from dataclasses import asdict, fields
from typing import Any, cast

from r0.modules.priors.lib.causal_graph import CausalGraphPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.recursive_forest_plan import recursive_effective_limits
from r0.modules.priors.lib.representatives import encode
from r0.modules.priors.lib.sampled_topology_hyperprior import SampledTopologyHyperprior
from r0.modules.priors.lib.topology_hyperprior import (
    FixedSizeTopologyHyperprior,
    TopologyHyperprior,
)
from r0.modules.priors.lib.topology_hyperprior_admission import admit_topology_hyperprior

_DEFAULT_LIMITS = CausalLimits()


def complete_topology_record(data: Any, kind: type[Any]) -> dict[str, Any]:
    raw = fitted_object(data, kind)
    if set(raw) != {f.name for f in fields(kind)}:
        raise ValueError("topology hyperprior archive requires every exact field")
    return raw


def topology_hyperprior_from_dict(data: Any):
    try:
        kind = (
            FixedSizeTopologyHyperprior
            if type(data) is dict
            and cast(dict[str, Any], data).get("version") == "fixed-size-topology-hyperprior-r0-v1"
            else TopologyHyperprior
        )
        return kind(**complete_topology_record(data, kind))
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete topology hyperprior request") from exc


def sampled_topology_hyperprior_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> SampledTopologyHyperprior:
    try:
        admit_function_metadata(data, limits)
        raw = complete_topology_record(data, SampledTopologyHyperprior)
        request = topology_hyperprior_from_dict(raw["request"])
        declared = CausalLimits(**complete_topology_record(raw["limits"], CausalLimits))
        raw_limits = RawDrawLimits(**complete_topology_record(raw["raw_limits"], RawDrawLimits))
        effective = recursive_effective_limits(declared, limits)
        budget = admit_topology_hyperprior(request, effective, raw_limits)
        topology_raw = complete_topology_record(raw["topology"], CausalGraphPrior)
        topology_raw["layer_widths"] = tuple(fitted_array(topology_raw["layer_widths"], 0))
        topology = CausalGraphPrior(**topology_raw)
        if not request.minimum_nodes <= topology.nodes <= request.maximum_nodes:
            raise ValueError("retained topology size is outside its declared range")
        records: list[RawDrawRecord] = []
        for item in fitted_array(raw["raw"], budget["draws"]):
            record = complete_topology_record(item, RawDrawRecord)
            for key in ("owner", "shape", "values"):
                record[key] = tuple(fitted_array(record[key], 16 if key == "owner" else 1))
            record["parameters"] = tuple(
                tuple(fitted_array(pair, 2)) for pair in fitted_array(record["parameters"], 2)
            )
            records.append(RawDrawRecord(**record))
        result = SampledTopologyHyperprior(
            request,
            raw["log_nodes"],
            raw["gamma_unit"],
            raw["gamma_value"],
            topology,
            tuple(
                tuple(fitted_array(v, 2))
                for v in fitted_array(raw["components"], request.components)
            ),
            tuple(tuple(fitted_array(v, 1)) for v in fitted_array(raw["parents"], topology.nodes)),
            tuple(records),
            GeneratorSnapshot(**complete_topology_record(raw["initial"], GeneratorSnapshot)),
            GeneratorSnapshot(**complete_topology_record(raw["final"], GeneratorSnapshot)),
            declared,
            raw_limits,
            raw["algorithm"],
            raw["version"],
        )
        result.replay(limits=effective)
        if encode(asdict(result)) != encode(data):
            raise ValueError("topology hyperprior archive differs from complete canonical record")
        return result
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError, IndexError) as exc:
        raise ValueError("invalid complete sampled topology hyperprior archive") from exc
