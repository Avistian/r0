"""Pinned, separately provisioned TabPFN contract; no implicit installation or assets."""

from .storage import Document

TABPFN: Document = {
    "package": "tabpfn==2.2.1",
    "wheel_sha256": "9aa3a0df8ecebcae5f426b1fec9bb2e2a690880bc8ad079191e8d959bf0f157b",
    "weights": {
        "classifier": "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49",
        "regressor": "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736",
    },
    "n_estimators": 8,
    "random_state": 0,
    "device": "cpu",
    "inference_precision": "float32",
    "n_preprocessing_jobs": 1,
    "fit_mode": "fit_preprocessors",
    "context_subsampling": None,
    "softmax_temperature": 0.9,
    "balance_probabilities": False,
    "average_before_softmax": False,
    "telemetry": "TABPFN_DISABLE_TELEMETRY=1",
    "preprocessing_recipes": "tabpfn_recipe.json: verified wheel defaults",
    "compatibility": "NOT_RUN",
    "query_invariance": "NOT_RUN",
    "environment": "separately_provisioned_required",
    "output": "probabilities_or_predictive_mean",
}


def tabpfn_status() -> Document:
    return {
        "status": "UNAVAILABLE",
        "recipe": TABPFN,
        "reason": "real execution requires separately provisioned assets and guarded lane; "
        "no installation, download or real inference authorized",
    }


def tabpfn_recipe(task: str) -> Document:
    """Explicit 2.2.1 defaults transcribed from the hash-verified release wheel."""
    from pathlib import Path

    from .storage import read_json

    if task not in ("binary", "multiclass", "regression"):
        raise ValueError("unknown task")
    key = "regressor" if task == "regression" else "classifier"
    parameters: Document = {
        "n_estimators": 8,
        "random_state": 0,
        "device": "cpu",
        "inference_precision": "float32",
        "n_jobs": 1,
        "fit_mode": "fit_preprocessors",
        "softmax_temperature": 0.9,
        "average_before_softmax": False,
        "ignore_pretraining_limits": False,
        "memory_saving_mode": "auto",
    }
    if key == "classifier":
        parameters.update(balance_probabilities=False, differentiable_input=False)
    return {
        "parameters": parameters,
        "inference_config": read_json(Path(__file__).with_name("tabpfn_recipe.json"))[key],
        "wheel_sha256": TABPFN["wheel_sha256"],
        "weight_sha256": TABPFN["weights"][key],
    }
