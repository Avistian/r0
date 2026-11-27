"""Check final artifacts, produce totals and SHA256 manifest without self-reference."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
components = {}
for name in ("coverage", "adversarial", "sharing", "gates"):
    result = json.loads((ROOT / "results" / (name + ".json")).read_text())
    components[name] = result["counts"]
    for details in result["imports"].values():
        filename = Path(details["path"])
        assert filename.is_relative_to(ROOT / "source"), filename
        assert hashlib.sha256(filename.read_bytes()).hexdigest() == details["sha256"]
assert not list(ROOT.rglob("*.pyc")), "Unexpected bytecode"
programs = {p.name: p.stat().st_size for p in (ROOT / "programs").glob("*.py")}
assert all(size <= 20_000 for size in programs.values())
legacy = json.loads((ROOT / "results/legacy-source/totals.json").read_text())
assert legacy["byte_equal_cases"] == 162
for name in ("ruff-final.log", "format-final.log"):
    assert (ROOT / "results" / name).is_file()
totals = {
    "source": "10d9b6c55107ff55c7e710f9adc01f96fc3c491b",
    "base": "1b214e44215e48ce964ea28818b1c5a14c9c14ca",
    "verdict": "PASS_COMPONENT",
    "components": components,
    "assertions": sum(c["assertions"] for c in components.values()),
    "expected_rejections": sum(c["rejections"] for c in components.values()),
    "legacy": legacy,
    "program_sizes": programs,
    "bytecode_files": 0,
    "graph_wide_acceptance": "PENDING",
}
(ROOT / "TOTALS.json").write_text(json.dumps(totals, indent=2) + "\n")
files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
manifest = "".join(
    f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n" for p in files
)
(ROOT / "MANIFEST.sha256").write_text(manifest)
print(
    json.dumps(
        {
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "assertions": totals["assertions"],
            "rejections": totals["expected_rejections"],
            "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        }
    )
)
