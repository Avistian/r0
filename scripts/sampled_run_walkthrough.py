"""Exercise saved sampled tasks, resume and explorer replay through public interfaces."""

import argparse
import base64
import hashlib
import io
import json
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch

from r0.modules.explorer import ExplorerSession
from r0.modules.generation import load_group, load_request, run_guarded
from r0.modules.priors import explain_generation, generate_task, recipe_from_dict


def main(out: Path) -> None:
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], check=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    out.mkdir(parents=True, exist_ok=False)
    config = Path("configs/generation/profile-polynomial-cpu.toml")
    request = load_request(config)
    (out / "request.toml").write_bytes(config.read_bytes())
    first = run_guarded(request, out / "run", workers=2)
    (out / "first.json").write_text(json.dumps(asdict(first), indent=2) + "\n")
    assert first.returncode == 0, first
    assert first.summary is not None
    assert first.summary["completed"] == request.count
    resumed = run_guarded(request, out / "run", workers=1)
    (out / "resumed.json").write_text(json.dumps(asdict(resumed), indent=2) + "\n")
    assert resumed.returncode == 0, resumed
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0
    assert resumed.summary["completed"] == request.count
    results = []
    for archive in sorted((out / "run").rglob("data-*.zip")):
        for task in load_group(archive):
            record = json.loads(task.recipe_json)
            recipe = recipe_from_dict(record["recipe"])
            # Replay from saved concrete settings, without consulting the profile sampler.
            replay = generate_task(recipe, seed=record["seed"], task_id=record["task_id"])
            assert task.episode() == replay.episode()
            for field in ("inputs", "scores", "targets"):
                assert torch.equal(getattr(task, field), getattr(replay, field))
            graph = explain_generation(task, [0, 1, 127])
            assert graph["target_max_error"] <= 1e-10
            results.append(
                {
                    "task_id": record["task_id"],
                    "recipe": record["recipe"],
                    "sampling": record["sampling"],
                    "episode_id": task.episode_id,
                    "graph_error": graph["target_max_error"],
                    "replay": "PASS",
                }
            )
    assert len(results) == request.count
    session = ExplorerSession(out / "explorer")
    original = session.request(
        "generate",
        {
            "record": {"recipe": {"rows": 32, "context_rows": 20}, "seed": 27, "task_id": 9},
            "sampling": {"family": "polynomial", "profile": "broad-v1"},
        },
    )
    independent = ExplorerSession(out / "replay")
    replayed = independent.request("generate", {"record_json": original["recipe_json"]})
    assert replayed["recipe_json"] == original["recipe_json"]
    assert replayed["run"]["summary"]["generated"] == 1
    assert session.request("row", {"index": 31}) == independent.request("row", {"index": 31})
    exported = base64.b64decode(session.request("export", {})["data"])
    (out / "replay-bundle.zip").write_bytes(exported)
    with zipfile.ZipFile(io.BytesIO(exported)) as bundle:
        export_entries = bundle.namelist()
    report = {
        "status": "PASS",
        "source": source,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tasks": results,
        "fresh_explorer_replay": "PASS",
        "export_entries": export_entries,
        "hosted_colab": "NOT_RUN",
        "modal": "NOT_RUN",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "tasks": len(results), "source": source}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    main(parser.parse_args().out)
