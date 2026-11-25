"""Independently check a small consumption receipt with NumPy and finite differences."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from r0.modules.generation import load_group
from r0.modules.priors import generate_task, recipe_from_dict


def calculation(episode, parameter):
    # Independently fit population scales on context only, without featurize.transform.
    x = np.asarray([row.values for row in episode.context], dtype=np.float64)
    q = np.asarray([row.values for row in episode.queries], dtype=np.float64)
    scale = np.std(x, axis=0)
    scale[scale == 0] = 1
    distance = np.mean(((q[:, None, :] - x[None, :, :]) / scale) ** 2, axis=2)
    logits = -distance / math.exp(parameter)
    weight = np.exp(logits - logits.max(axis=1, keepdims=True))
    weight /= weight.sum(axis=1, keepdims=True)
    if episode.task_type == "regression":
        prediction = weight @ np.asarray(episode.context_targets)
        loss = np.mean((prediction - np.asarray(episode.query_targets)) ** 2)
    else:
        # Sum weight by label, independently of Torch's one-hot matrix multiplication.
        labels = np.asarray(episode.context_targets)
        prediction = np.column_stack(
            [
                0.99 * weight[:, labels == label].sum(axis=1) + 0.01 / len(episode.classes)
                for label in episode.classes
            ]
        )
        truth = [episode.classes.index(str(label)) for label in episode.query_targets]
        loss = -np.log(prediction[np.arange(len(truth)), truth]).mean()
    return prediction, float(loss)


def main(directory: Path, archive: Path | None, output: Path):
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "COMPLETE"
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    # An evidence-only commit is harmless, but changed generation/featurization is not.
    assert (
        subprocess.run(
            ["git", "diff", "--quiet", summary["source"], "--", "src"], check=False
        ).returncode
        == 0
    )
    saved = None
    if summary["archive"] is not None:
        archive = archive or Path(summary["archive"])
        with archive.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == summary["archive_sha256"]
        saved = iter(load_group(archive))
    identities = set()
    errors = dict(prediction=0.0, loss=0.0, gradient=0.0, update=0.0)
    batches, episodes, max_rows, max_cells = 0, 0, 0, 0
    sizes, coverage, absent = [], set(), 0
    parameter = 0.0
    h = 1e-5
    with (directory / "batches.jsonl").open() as stream:
        for line in stream:
            report = json.loads(line)
            assert report["parameter_before"] == parameter
            losses, plus, minus = [], [], []
            rows, cells = 0, 0
            with np.load(directory / report["prediction_file"], allow_pickle=False) as arrays:
                assert set(arrays.files) == {r["predictions"] for r in report["episodes"]}
                for record in report["episodes"]:
                    replay = record["recipe"]
                    task = (
                        next(saved)
                        if saved is not None
                        else generate_task(
                            recipe_from_dict(replay["recipe"]),
                            seed=replay["seed"],
                            task_id=replay["task_id"],
                        )
                    )
                    assert task.source_task_id == record["source_task_id"]
                    assert task.episode_id == record["episode_id"]
                    assert json.loads(task.recipe_json) == replay
                    assert task.source_task_id not in identities
                    identities.add(task.source_task_id)
                    episode = task.episode()
                    prediction, loss = calculation(episode, parameter)
                    retained = arrays[record["predictions"]]
                    np.testing.assert_allclose(retained, prediction, rtol=1e-11, atol=1e-12)
                    np.testing.assert_allclose(loss, record["loss"], rtol=1e-11, atol=1e-12)
                    errors["prediction"] = max(
                        errors["prediction"], float(abs(retained - prediction).max())
                    )
                    errors["loss"] = max(errors["loss"], abs(loss - record["loss"]))
                    losses.append(loss)
                    plus.append(calculation(episode, parameter + h)[1])
                    minus.append(calculation(episode, parameter - h)[1])
                    assert record["query_target_counterfactual"] == "PASS"
                    n = len(episode.context) + len(episode.queries)
                    assert record["rows"] == n
                    assert record["features"] == len(episode.features)
                    assert record["context_rows"] == len(episode.context)
                    assert record["query_rows"] == len(episode.queries)
                    assert record["task_type"] == episode.task_type
                    assert record["classes"] == len(episode.classes)
                    missing = len(set(episode.classes) - set(map(str, episode.context_targets)))
                    assert missing == record["missing_context_classes"]
                    absent += int(missing > 0)
                    rows += n
                    cells += len(episode.context) * len(episode.queries)
                    coverage.add((record["family"], episode.task_type))
                    episodes += 1
            size = len(losses)
            assert 0 < size <= summary["batch_size"]
            sizes.append(size)
            gradient = (float(np.mean(plus)) - float(np.mean(minus))) / (2 * h)
            np.testing.assert_allclose(np.mean(losses), report["objective"], rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(gradient, report["gradient"], rtol=1e-6, atol=1e-8)
            update = parameter - summary["learning_rate"] * report["gradient"]
            np.testing.assert_allclose(update, report["parameter_after"], rtol=1e-12, atol=1e-14)
            errors["gradient"] = max(errors["gradient"], abs(gradient - report["gradient"]))
            errors["update"] = max(errors["update"], abs(update - report["parameter_after"]))
            parameter = report["parameter_after"]
            max_rows, max_cells = max(max_rows, rows), max(max_cells, cells)
            batches += 1
    assert all(n == summary["batch_size"] for n in sizes[:-1])
    assert episodes == summary["consumed"] == summary["produced"]
    assert batches == summary["batches"]
    assert max(sizes) == summary["max_batch_episodes"]
    assert max_rows == summary["max_batch_rows"]
    assert max_cells == summary["max_batch_distance_cells"]
    assert parameter == summary["final_log_bandwidth"]
    assert episodes <= summary["requested_episodes"]
    if episodes < summary["requested_episodes"]:
        assert saved is not None
        assert next(saved, None) is None, "saved source stopped before archive exhaustion"
    result = {
        "status": "PASS",
        "source": source,
        "consumption_source": summary["source"],
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "batches": batches,
        "episodes": episodes,
        "batch_sizes": sizes,
        "family_task_type_coverage": sorted(coverage),
        "episodes_with_absent_context_classes": absent,
        "max_absolute_errors": errors,
        "finite_difference_step": h,
        "prediction_tolerance": {"rtol": 1e-11, "atol": 1e-12},
        "gradient_tolerance": {"rtol": 1e-6, "atol": 1e-8},
    }
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.directory, args.archive, args.output)
