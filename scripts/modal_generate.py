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
        .env(
            {
                "PYTHONPATH": "/opt/r0",
                "R0_GENERATION_SOURCE": source,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        .add_local_file("scripts/check_device_agreement.py", "/opt/check_device_agreement.py")
        .add_local_dir("src/r0", remote_path="/opt/r0/r0", ignore=["**/__pycache__/**"])
    )

    @app.function(
        image=image,
        serialized=True,
        volumes={"/outputs": volume},
        cpu=(1, 1),
        memory=(4096, 4096),
        gpu=None if plan["gpu"] == "none" else "T4",
        timeout=int(plan["request"]["max_seconds"]) + 60 + plan["agreement_seconds"],
        startup_timeout=120,
        retries=0,
        max_containers=1,
        single_use_containers=True,
    )
    def execute(request_plan, phase):
        import json
        import os
        import subprocess
        import sys
        from dataclasses import asdict
        from pathlib import Path

        from r0.modules.generation import request_from_dict, run_guarded

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
        request = request_from_dict(request_plan["request"])
        out = root / "runs" / request_plan["run"]
        try:
            agreement = "NOT_RUN"
            if request.device == "cuda" and phase == "generate":
                agreement_dir = receipt_dir / f"{name}-agreement"
                agreement_dir.mkdir(exist_ok=False)
                plan_path = agreement_dir / "request-plan.json"
                plan_path.write_text(json.dumps(request_plan))
                comparison_dir = agreement_dir / "comparison"
                timed_out = False
                with (agreement_dir / "command.log").open("x") as log:
                    try:
                        completed = subprocess.run(
                            [
                                sys.executable,
                                "/opt/check_device_agreement.py",
                                "--plan",
                                str(plan_path),
                                "--out",
                                str(comparison_dir),
                                "--candidate",
                                "cuda",
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=request_plan["agreement_seconds"],
                            check=False,
                        )
                        returncode = completed.returncode
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        returncode = 124
                records = {
                    path.name: json.loads(path.read_text())
                    for path in comparison_dir.glob("*.json")
                }
                summary = records.get("summary.json", {})
                agreement = {
                    "status": "TIMEOUT" if timed_out else summary.get("status", "INCOMPLETE"),
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "records": records,
                    "log": (agreement_dir / "command.log").read_text(),
                }
                if returncode != 0 or agreement["status"] != "PASS":
                    report = {
                        "returncode": returncode or 1,
                        "timed_out": timed_out,
                        "summary": {"generated": 0},
                        "source": os.environ["R0_GENERATION_SOURCE"],
                        "device_agreement": agreement,
                    }
                    receipt.write_text(json.dumps(report))
                    return report
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
