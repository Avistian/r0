"""Execute notebook smoke/resume cells locally; skip install, Drive and larger runs."""

import argparse
import ast
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import r0
from r0.modules.generation import verify_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", default="Linear")
    parser.add_argument("--profile", default="broad-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert Path(r0.__file__).resolve().is_relative_to(root / "src")
    notebook = json.loads((root / "notebooks/generation.ipynb").read_text())
    cells = ["".join(cell["source"]) for cell in notebook["cells"]]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<notebook-cell>", "exec")
    revision = next(
        ast.literal_eval(statement.value)
        for statement in ast.parse(cells[1]).body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "REVISION" for target in statement.targets
        )
    )
    subprocess.run(["git", "diff", "--exit-code", revision, "--", "src"], cwd=root, check=True)
    assert not args.out.exists(), "use a fresh walkthrough output directory"
    args.out.mkdir(parents=True)
    namespace = {"ROOT": args.out, "REVISION": revision}
    assert cells[4].count('FAMILY = "Linear"') == 1
    generation = cells[4].replace('FAMILY = "Linear"', f"FAMILY = {args.family!r}")
    assert generation.count('PROFILE = "broad-v1"') == 1
    generation = generation.replace('PROFILE = "broad-v1"', f"PROFILE = {args.profile!r}")
    exec(compile(generation, "<generation-cell>", "exec"), namespace)
    first = asdict(namespace["smoke"])
    assert first["summary"]["generated"] == 8
    assert namespace["measurement_path"].exists(), "timing must survive before estimate cell"
    namespace = {"ROOT": args.out, "REVISION": revision}
    exec(compile(generation, "<resume-cell>", "exec"), namespace)
    resumed = asdict(namespace["smoke"])
    assert resumed["summary"]["generated"] == 0
    assert resumed["summary"]["completed"] == 8
    exec(compile(cells[6], "<retained-estimate-cell>", "exec"), namespace)
    exec(compile(cells[7], "<larger-run-disabled>", "exec"), namespace)
    assert "large" not in namespace
    exec(compile(cells[9], "<verify-cell>", "exec"), namespace)
    receipts = [asdict(verify_group(path)) for path in namespace["archives"]]
    report = {
        "scope": "local notebook generation cells, not hosted Colab/UI/Drive proof",
        "pinned_source": revision,
        "family": args.family,
        "profile": args.profile,
        "first": first,
        "resume": resumed,
        "groups": receipts,
        "larger_run": "NOT_RUN",
    }
    (args.out / "walkthrough.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
