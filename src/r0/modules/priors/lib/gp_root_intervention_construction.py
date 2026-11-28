"""Admit raw GP intervention controls and exact identity before decoding either graph."""

from typing import Any

from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_root_intervention import GPRootIntervention, intervene_gp_root_dag
from r0.modules.priors.lib.gp_root_intervention_decoding import decode_gp_intervention_source
from r0.modules.priors.lib.gp_root_intervention_metadata import intervention_plan


def construct_gp_root_intervention(
    data: dict[str, Any], *, seed: int, task_id: int, limits: CausalLimits
) -> GPRootIntervention:
    if type(data) is not dict or set(data) != {"source", "change"}:
        raise ValueError("GP intervention construction requires complete source and change")
    if any(type(value) is not int or not 0 <= value < 2**63 for value in (seed, task_id)):
        raise ValueError("GP intervention requires exact original identity")
    metadata, _, _, _ = intervention_plan(data["source"], data["change"], limits=limits)
    current = metadata.current["source"]
    if (seed, task_id) != (current["seed"], current["task_id"]):
        raise ValueError("GP intervention construction identity differs from original source")
    source = decode_gp_intervention_source(data["source"], limits=metadata.limits)
    change = intervention_from_dict(data["change"], limits=metadata.limits)
    return intervene_gp_root_dag(source, change, limits=metadata.limits)
