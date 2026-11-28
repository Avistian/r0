"""Cheap calibrated saved-task transport checks, before any retained stream replay."""

from typing import Any, cast

from r0.modules.priors.lib.population_generation_identity import preflight_calibrated_generation
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def preflight_calibrated_inspection(task: TrainingTask, record: dict[str, Any]) -> None:
    """Bind exact outer identity, execution and split linkage to serialized source metadata.

    The caller first validates the complete population record schema. Historical
    library/environment descriptions remain provenance, not current-version gates.
    Full recipe and tensor validation still follows this metadata-only admission.
    """
    source = record["recipe"]
    execution = record["execution"]
    preflight_calibrated_generation(
        source, seed=record["seed"], task_id=record["task_id"], device=execution["device"]
    )
    if execution["implementation"] != source["version"]:
        raise ValueError("calibrated inspection implementation differs from retained source")
    original = (
        source["base"]["original"]
        if source["version"] == "calibrated-population-intervention-v1"
        else source["original"]
    )
    request = original.get("request")
    base = cast(dict[str, Any], request).get("base") if type(request) is dict else None
    geometry = cast(dict[str, Any], base).get("task") if type(base) is dict else None
    if type(geometry) is not dict:
        raise ValueError("calibrated inspection requires complete task metadata")
    geometry = cast(dict[str, Any], geometry)
    split, context = geometry.get("split"), geometry.get("context_rows")
    if type(split) is not int or not 0 <= split < 2**63 or type(context) is not int:
        raise ValueError("calibrated inspection requires exact split metadata")
    source_id = record_fingerprint((source, record["seed"], record["task_id"]))
    episode_id = record_fingerprint((source_id, split, context))
    if (
        type(task.source_task_id) is not str
        or task.source_task_id != source_id
        or type(task.episode_id) is not str
        or task.episode_id != episode_id
    ):
        raise ValueError(
            "calibrated inspection source or episode semantic linkage differs from record"
        )
