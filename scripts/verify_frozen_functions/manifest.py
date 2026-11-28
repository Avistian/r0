"""Bind frozen source, final executed verifiers, retained outcomes and imports."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
files = {}
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        files[str(path.relative_to(ROOT))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
counts = {
    "check_ties": 98,
    "check_roundtrip": 1,
    "check_equations": 142,
    "reproduce_quasi_tie": 1,
    "check_contracts": 157,
    "check_admission": 291,
    "check_legacy": 258,
}
for stem in (
    "check_equations",
    "reproduce_quasi_tie",
    "check_contracts",
    "check_admission",
    "check_legacy",
):
    receipt = json.loads((ROOT / f"{stem}-receipt.json").read_text())
    assert receipt["cases"] == counts[stem]
    for name in (f"{stem}.py", "evidence_common.py"):
        assert receipt["programs"][name] == files[name]["sha256"]
    events = [json.loads(line) for line in (ROOT / f"{stem}.jsonl").read_text().splitlines()]
    assert len(events) == counts[stem]
    assert all(event["status"] == "PASS" for event in events)
    for imported in receipt["imports"].values():
        path = Path(imported["path"])
        assert path.is_relative_to(ROOT / "source")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == imported["sha256"]
assert (
    json.loads((ROOT / "roundtrip.json").read_text())["program_sha256"]
    == files["check_roundtrip.py"]["sha256"]
)
for revision in ("base", "source"):
    receipt = json.loads((ROOT / f"r0-{revision}" / "receipt.json").read_text())
    assert receipt["program_sha256"] == files["check_r0_regression.py"]["sha256"]
    assert receipt["completed"] == 18
    assert receipt["failed"] == 0
    if revision == "source":
        assert receipt["byte_equal_cases"] == 18
    for imported in receipt["imports"].values():
        path = Path(imported["path"])
        assert path.is_relative_to(ROOT / revision)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == imported["sha256"]
previous = Path("/tmp/r0-28-functions-spec-ed5b67a")
for name in (
    "check_ties.py",
    "check_roundtrip.py",
    "check_equations.py",
    "evidence_common.py",
    "reproduce_quasi_tie.py",
    "I2func.py",
):
    assert hashlib.sha256((previous / name).read_bytes()).hexdigest() == files[name]["sha256"]
for path in ROOT.glob("*.py"):
    if path.name != "I2func.py":
        assert path.stat().st_size <= 20000
manifest = {
    "source": "cbbaff2cb4c8c6e485ad3814f79e84f3c1d68967",
    "base": "cb7977a916ff6f11d95790b2de4dd591ab8d3f39",
    "verdict": "PASS_COMPONENT",
    "case_counts": counts,
    "total_final_cases": sum(counts.values()),
    "prior_r0_paired_comparisons": 18,
    "prior_r0_completed_executions": 36,
    "numerical_harness_failures": 0,
    "product_failures": 0,
    "files": files,
}
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"PASS_COMPONENT: {sum(counts.values())} final cases, {len(files)} exact file identities.")
