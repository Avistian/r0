"""Saved task groups are inspected as ordinary archives, independently of the writer."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from r0.modules.generation import save_group, verify_group
from r0.modules.priors import LinearRecipe, generate_linear


def test_group_separates_model_inputs_from_truth_and_detects_corruption(tmp_path: Path) -> None:
    task = generate_linear(LinearRecipe(), seed=5, task_id=3)
    path = tmp_path / "group.zip"
    receipt = save_group(path, [(3, task)], max_bytes=1_000_000)
    assert verify_group(path) == receipt
    assert receipt.task_ids == (3,)
    with zipfile.ZipFile(path) as archive:
        with np.load(io.BytesIO(archive.read("3/predictor.npz")), allow_pickle=False) as data:
            assert set(data.files) == {"context_x", "context_y", "query_x"}
            assert data["query_x"].shape == (16, 8)
        with np.load(io.BytesIO(archive.read("3/truth.npz")), allow_pickle=False) as data:
            assert data["query_y"].shape == (16,)
            assert data["scores"].shape == (64, 1)
        card = json.loads(archive.read("3/card.json"))
        assert card["source_task_id"] == task.source_task_id
        assert card["recipe"]["task_id"] == 3
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="checksum"):
        verify_group(path)


def test_interrupted_group_is_not_complete_and_existing_output_is_not_overwritten(
    tmp_path: Path,
) -> None:
    task = generate_linear(LinearRecipe(), seed=5, task_id=3)
    path = tmp_path / "limited.zip"
    with pytest.raises(ValueError, match="byte limit"):
        save_group(path, [(3, task)], max_bytes=4096)
    assert path.stat().st_size <= 4096
    with pytest.raises(ValueError, match="no completion"):
        verify_group(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_group(path, [(3, task)], max_bytes=1_000_000)
    assert path.read_bytes() == original


def test_group_writer_rejects_duplicate_task_id_without_completing(tmp_path: Path) -> None:
    task = generate_linear(LinearRecipe(), seed=5, task_id=3)
    path = tmp_path / "duplicate.zip"
    with pytest.raises(ValueError, match="unique"):
        save_group(path, [(3, task), (3, task)], max_bytes=1_000_000)
    with pytest.raises(ValueError, match="no completion"):
        verify_group(path)
