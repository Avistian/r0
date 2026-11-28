"""Validate complete retained GP-root task identity, semantics and tensors."""

import json

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration, generate_calibrated_gp_root
from r0.modules.priors.lib.gp_root_calibration_decoding import gp_root_calibration_from_dict
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe, generate_gp_root_dag
from r0.modules.priors.lib.gp_root_decoding import gp_root_dag_from_dict
from r0.modules.priors.lib.gp_task_validation import validate_gp_execution, validate_gp_task
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def validate_gp_root_task(
    task: TrainingTask,
    *,
    limits: CausalLimits | None = None,
) -> tuple[GPRootDAGRecipe | GPRootCalibration, TrainingTask]:
    """Return the admitted recipe and canonical replay after complete validation."""
    try:
        record = json.loads(task.recipe_json)
        raw = record["recipe"]
        calibrated = raw.get("version") == "gp-root-calibration-v1"
        source = raw["original"]["source"] if calibrated else raw["source"]
        if set(record) != {"recipe", "seed", "task_id", "execution"} or any(
            type(record[k]) is not int or record[k] != source[k] for k in ("seed", "task_id")
        ):
            raise ValueError("GP root task transport identity differs")
        version = "gp-root-calibration-v1" if calibrated else "gp-root-dag-v1"
        validate_gp_execution(record["execution"], version)
        controls = raw["original"]["base"]["task"] if calibrated else raw["base"]["task"]
        source_id = record_fingerprint((raw, record["seed"], record["task_id"]))
        episode_id = record_fingerprint((source_id, controls["split"], controls["context_rows"]))
        if (
            task.source_task_id != source_id
            or task.episode_id != episode_id
            or task.task_type != controls["task_type"]
        ):
            raise ValueError("GP source, episode or task metadata differs from retained recipe")

        if calibrated:
            if record["execution"]["device"] != "cpu":
                raise ValueError("GP calibration requires retained CPU execution")
            recipe = gp_root_calibration_from_dict(raw, limits=limits or CausalLimits())
            expected = generate_calibrated_gp_root(
                recipe, seed=record["seed"], task_id=record["task_id"]
            )
            validate_gp_task(task, expected, recipe.version)
            return recipe, expected
        recipe = gp_root_dag_from_dict(raw, limits=limits or CausalLimits())
        expected = generate_gp_root_dag(recipe, seed=record["seed"], task_id=record["task_id"])
        validate_gp_task(task, expected, recipe.version)
        return recipe, expected
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("malformed GP root task transport") from exc
