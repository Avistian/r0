"""Saved tasks remain usable by the shared training/inspection contract."""

import weakref
from pathlib import Path

import pytest
import torch

from r0.modules.generation import load_group, save_group
from r0.modules.priors import LinearRecipe, generate_task


def test_saved_group_reconstructs_tasks_and_episodes_without_generation(tmp_path: Path) -> None:
    tasks = [
        generate_task(recipe, seed=27, task_id=i)
        for i, recipe in enumerate(
            (
                LinearRecipe(),
                LinearRecipe(task_type="binary", classes=2),
                LinearRecipe(task_type="multiclass", classes=5),
            )
        )
    ]
    archive = tmp_path / "group.zip"
    save_group(archive, enumerate(tasks), max_bytes=1024**2)
    loaded = list(load_group(archive))
    assert len(loaded) == 3
    for original, restored in zip(tasks, loaded, strict=True):
        assert original.episode() == restored.episode()
        assert original.recipe_json == restored.recipe_json
        assert original.mechanism_json == restored.mechanism_json
        for field in ("inputs", "targets", "scores", "context_indices", "query_indices"):
            torch.testing.assert_close(
                getattr(original, field), getattr(restored, field), rtol=0, atol=0
            )
        if original.probabilities is not None:
            torch.testing.assert_close(
                original.probabilities, restored.probabilities, rtol=0, atol=0
            )
    with archive.open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        next(load_group(archive))


def test_loader_does_not_retain_the_task_after_consumer_releases_it(tmp_path: Path) -> None:
    archive = tmp_path / "group.zip"
    save_group(
        archive,
        ((i, generate_task(LinearRecipe(), seed=27, task_id=i)) for i in range(2)),
        max_bytes=1024**2,
    )
    stream = load_group(archive)
    first = next(stream)
    inputs = weakref.ref(first.inputs)
    del first
    assert inputs() is None
    assert next(stream).source_task_id
