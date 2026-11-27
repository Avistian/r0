"""Run independently authored public-API checks against the pinned component."""

import hashlib
import traceback

from ancestry_cases import verify_ancestry, verify_calibration
from common import BASE, HEAD, OUT, checks, custody, save
from contract_cases import verify_moments, verify_records
from pipeline_cases import verify_pipelines
from root_cases import verify_roots


def main():
    idx = verify_roots()
    act, root, pipe, child = verify_pipelines(idx)
    verify_moments(act, root, pipe, child)
    verify_records()
    verify_ancestry()
    verify_calibration(root)
    save(
        "summary.json",
        dict(
            status="PASS",
            checks=len(checks),
            head=HEAD,
            baseline=BASE,
            scope="authored root input plus calibrated ACT converters",
            limitations=[
                "Exact PRNG replay is not statistical proof of stream independence",
                "Independent NumPy equations and Student-t mathematical moments; "
                "not source RNG parity",
                "CPU API explorer workflows only; "
                "browser/Colab/GPU/throughput/scientific comparison NOT_RUN",
            ],
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        save(
            "failure.json",
            dict(
                status="REJECT_OR_HARNESS_ERROR",
                traceback=traceback.format_exc(),
                last_check=checks[-1] if checks else None,
            ),
        )
        raise
    finally:
        custody("custody-after.json")
        save(
            "manifest.json",
            {
                str(f.relative_to(OUT)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(OUT.rglob("*"))
                if f.is_file() and f.name != "manifest.json"
            },
        )
