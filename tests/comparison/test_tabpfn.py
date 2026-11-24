"""Pinned local asset checks do not load a model or access the network."""

from pathlib import Path

import pytest

from r0.modules import comparison


def test_tabpfn_rejects_wrong_wheel_before_import(tmp_path: Path) -> None:
    wheel = tmp_path / "wrong.whl"
    wheel.write_bytes(b"not the pinned release")
    model = tmp_path / "model.ckpt"
    model.write_bytes(b"fixture, not weights")
    validate = comparison.validate_tabpfn_assets
    with pytest.raises(ValueError, match="wheel hash"):
        validate(wheel, model, "binary")


def test_recipe_is_resolved_for_both_task_types() -> None:
    recipe = comparison.tabpfn_recipe
    classification = recipe("binary")
    regression = recipe("regression")
    assert (
        classification["inference_config"]["PREPROCESS_TRANSFORMS"][0]["name"]
        == "quantile_uni_coarse"
    )
    assert regression["inference_config"]["PREPROCESS_TRANSFORMS"][1]["name"] == "safepower"
    assert regression["inference_config"]["SUBSAMPLE_SAMPLES"] is None
    assert classification["parameters"]["n_estimators"] == 8


def test_tabpfn_recipe_does_not_allow_mutating_later_requests() -> None:
    first = comparison.tabpfn_recipe("binary")
    first["inference_config"]["SUBSAMPLE_SAMPLES"] = 1
    assert comparison.tabpfn_recipe("binary")["inference_config"]["SUBSAMPLE_SAMPLES"] is None
