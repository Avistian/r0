"""Typed observation and explicit missing-cell transport without latent leakage."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def observed() -> p.TrainingTask:
    task = p.generate_task(
        p.LinearRecipe(rows=16, context_rows=8, features=3, signal_features=3, active_features=3),
        seed=30,
        task_id=7,
    )
    inputs = torch.tensor(
        [
            [0, 0, 0],
            [2, 1, 0],
            [3, 2, 0],
            [4, 0, 0],
            [5, 1, 0],
            [6, 2, 0],
            [7, 0, 0],
            [8, 1, 0],
        ],
        dtype=torch.float64,
    )
    inputs = inputs.repeat(2, 1)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    mask[0, 0] = mask[3, 1] = mask[6, 2] = True
    inputs[mask] = float("nan")
    return replace(
        task,
        inputs=inputs,
        missing_mask=mask,
        feature_schema=(
            p.TrainingFeature("count"),
            p.TrainingFeature("ordinal", ("low", "medium", "high")),
            p.TrainingFeature("categorical", ("singleton",)),
        ),
    )


def test_typed_missing_archive_and_episode(tmp_path: Path) -> None:
    task = observed()
    episode = task.episode()
    assert episode.features[1].order == ("low", "medium", "high")
    rows = {r.id: r.values for r in (*episode.context, *episode.queries)}
    assert rows["row-0"] == (None, "low", "singleton")
    assert rows["row-3"] == (4, None, "singleton")
    assert rows["row-6"] == (7, "low", None)
    path = tmp_path / "observed.zip"
    g.save_group(path, [(7, task)], max_bytes=2000000)
    saved = next(g.load_group(path))
    assert saved.feature_schema == task.feature_schema
    torch.testing.assert_close(saved.inputs, task.inputs, equal_nan=True, rtol=0, atol=0)
    assert saved.missing_mask is not None
    assert task.missing_mask is not None
    assert torch.equal(saved.missing_mask, task.missing_mask)
    assert saved.episode() == episode


@pytest.mark.parametrize("invalid", ["unmarked_nan", "marked_finite", "integer_mask", "infinity"])
def test_missing_mask_is_explicit_and_exact(invalid: str) -> None:
    task = observed()
    assert task.missing_mask is not None
    mask = task.missing_mask.clone()
    inputs = task.inputs.clone()
    if invalid == "unmarked_nan":
        mask[0, 0] = False
    elif invalid == "marked_finite":
        mask[1, 0] = True
    elif invalid == "integer_mask":
        mask = mask.to(torch.int64)
    else:
        inputs[1, 0] = float("inf")
    with pytest.raises(ValueError, match="missing mask"):
        replace(task, inputs=inputs, missing_mask=mask)


@pytest.mark.parametrize("value", [-1.0, 0.5, float("inf")])
def test_count_rejects_invalid_observed_values(value: float) -> None:
    with pytest.raises(ValueError, match="count|finite"):
        p.TrainingFeature("count").decode(value)


def test_ordinal_order_and_present_codes_are_validated() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        p.TrainingFeature("ordinal", ("same", "same"))
    task = observed()
    inputs = task.inputs.clone()
    inputs[1, 1] = 3
    with pytest.raises(ValueError, match="codes"):
        replace(task, inputs=inputs)


def test_all_missing_columns_remain_valid() -> None:
    task = observed()
    inputs = torch.full_like(task.inputs, float("nan"))
    mask = torch.ones_like(inputs, dtype=torch.bool)
    missing = replace(task, inputs=inputs, missing_mask=mask)
    episode = missing.episode()
    assert all(row.values == (None, None, None) for row in (*episode.context, *episode.queries))
