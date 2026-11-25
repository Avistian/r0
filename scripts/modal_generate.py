"""Credential-bearing adapter; invoke only from the guarded agent-train workflow."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if plan["source"] != source or plan["workflow_run_id"] != os.environ.get("GITHUB_RUN_ID"):
        raise ValueError("plan does not belong to this checkout and workflow invocation")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], check=True)

    import modal

    app = modal.App("r0-generation")
    volume = modal.Volume.from_name("r0-generation-v1", create_if_missing=True, version=2)
    image = (
        modal.Image.debian_slim(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
        .uv_pip_install("numpy==2.5.5", "torch==2.15.0")
        .env({"PYTHONPATH": "/opt/r0", "R0_GENERATION_SOURCE": source})
        .add_local_dir("src/r0", remote_path="/opt/r0/r0", ignore=["**/__pycache__/**"])
    )

    @app.function(
        image=image,
        serialized=True,
        volumes={"/outputs": volume},
        cpu=(1, 1),
        memory=(4096, 4096),
        gpu=None if plan["gpu"] == "none" else "T4",
        timeout=int(plan["request"]["max_seconds"]) + 60,
        startup_timeout=120,
        retries=0,
        max_containers=1,
        single_use_containers=True,
    )
    def execute(request_plan, phase):
        import json
        import os
        from dataclasses import asdict
        from pathlib import Path

        from r0.modules.generation import RunRequest, run_guarded
        from r0.modules.priors import LinearRecipe

        root = Path("/outputs")
        receipt_dir = root / "receipts"
        receipt_dir.mkdir(exist_ok=True)
        name = f"{request_plan['workflow_run_id']}-{phase}"
        receipt = receipt_dir / f"{name}.json"
        claim = receipt_dir / f"{name}.started"
        if receipt.exists():
            return json.loads(receipt.read_text())
        if claim.exists():
            raise RuntimeError("previous invocation has an uncertain outcome; no automatic retry")
        with claim.open("x") as handle:
            handle.write(json.dumps(request_plan))
        volume.commit()
        config = dict(request_plan["request"])
        recipe = config.pop("recipe")
        recipe["class_offsets"] = tuple(recipe["class_offsets"])
        request = RunRequest(LinearRecipe(**recipe), **config)
        out = root / "runs" / request_plan["run"]
        try:
            agreement = "NOT_RUN"
            if request.device == "cuda" and phase == "generate":
                from dataclasses import replace

                import torch

                from r0.modules.priors import generate_linear

                for task_type, classes in (("regression", 0), ("binary", 2), ("multiclass", 3)):
                    recipe = replace(
                        request.recipe, task_type=task_type, classes=classes, noise_ratio=0.0
                    )
                    cpu = generate_linear(recipe, seed=request.seed, task_id=request.start)
                    gpu = generate_linear(
                        recipe, seed=request.seed, task_id=request.start, device="cuda"
                    )
                    assert cpu.source_task_id == gpu.source_task_id
                    assert cpu.mechanism_json == gpu.mechanism_json
                    for field in (
                        "inputs",
                        "scores",
                        "targets",
                        "context_indices",
                        "query_indices",
                    ):
                        torch.testing.assert_close(
                            getattr(cpu, field), getattr(gpu, field).cpu(), rtol=1e-10, atol=1e-10
                        )
                    if cpu.probabilities is not None:
                        torch.testing.assert_close(
                            cpu.probabilities, gpu.probabilities.cpu(), rtol=1e-10, atol=1e-10
                        )
                agreement = "PASS: regression/binary/multiclass; rtol=atol=1e-10"
            result = run_guarded(
                request,
                out,
                workers=request_plan["workers"],
                retry_failed=request_plan["retry_failed"] if phase == "generate" else False,
                sync_path=root,
            )
            report = {
                **asdict(result),
                "source": os.environ["R0_GENERATION_SOURCE"],
                "device_agreement": agreement,
            }
            receipt.write_text(json.dumps(report))
        finally:
            # Persist both successes and interrupted output before returning to the caller.
            volume.commit()
        return report

    result = {"plan": plan, "status": "SUBMITTING", "actual_cost_usd": "NOT_MEASURED"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    try:
        with modal.enable_output(), app.run():
            first = execute.remote(plan, "generate")
            result["generation"] = first
            if first["returncode"] != 0 or first["timed_out"]:
                result["status"] = "INCOMPLETE"
                return 1
            # A separate container proves reloading/verification after provider commit.
            resumed = execute.remote(plan, "resume")
            result["resume"] = resumed
            if resumed["returncode"] != 0 or resumed["summary"]["generated"] != 0:
                result["status"] = "RESUME_FAILED"
                return 1
            result["status"] = "COMPLETE"
            return 0
    except BaseException as error:
        result["status"] = "FAILED_OR_UNCERTAIN"
        result["error"] = type(error).__name__
        raise
    finally:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
