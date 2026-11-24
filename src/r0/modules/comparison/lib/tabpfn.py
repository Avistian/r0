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
    "preprocessing_recipes": "PENDING_SOURCE_VERIFICATION",
    "compatibility": "NOT_RUN",
    "query_invariance": "NOT_RUN",
    "environment": "separately_provisioned_required",
    "output": "probabilities_or_predictive_mean",
}


def tabpfn_status() -> Document:
    return {
        "status": "UNAVAILABLE",
        "recipe": TABPFN,
        "reason": "real execution requires guarded lane and frozen resolved preprocessing; "
        "no installation, download or real inference authorized",
    }
