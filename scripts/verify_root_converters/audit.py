"""Compare retained legacy results and each loaded source against its pinned Git bytes."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("ROOT_CONVERTER_PACKAGE", Path(__file__).parent.parent)).resolve()
REPO = Path(os.environ.get("ROOT_CONVERTER_REPO", "/tmp/r0-28")).resolve()
BASE = "6ba0bc2813b777749c9944d93b404ef0348168c3"
HEAD = "c997bc2dc98db7611d7aa78552ee95467093301e"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = ROOT / "final-legacy-baseline"
    head = ROOT / "final-legacy-head"
    same = (baseline / "records.json").read_bytes() == (head / "records.json").read_bytes()
    receipts = []
    for folder, commit in [(baseline, BASE), (head, HEAD), (ROOT / "final", HEAD)]:
        custody = folder / ("custody.json" if folder != ROOT / "final" else "custody-after.json")
        record = json.loads(custody.read_text())
        for name, item in record["imports"].items():
            file = Path(item["path"])
            relative = "src/" + item["path"].split("/src/", 1)[1]
            git_bytes = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=REPO)
            expected = hashlib.sha256(git_bytes).hexdigest()
            receipts.append(
                dict(
                    run=folder.name,
                    module=name,
                    sha256=item["sha256"],
                    git_sha256=expected,
                    working_sha256=digest(file),
                    matched=item["sha256"] == expected == digest(file),
                )
            )
    summary = json.loads((head / "summary.json").read_text())
    result = dict(
        status="PASS" if same and all(r["matched"] for r in receipts) else "REJECT",
        head=HEAD,
        baseline=BASE,
        legacy_byte_equal=same,
        legacy=summary,
        module_checks=len(receipts),
        source_receipts=receipts,
    )
    (ROOT / "final-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        str(file.relative_to(ROOT)): digest(file)
        for file in ROOT.rglob("*")
        if file.is_file()
        and "baseline" not in file.relative_to(ROOT).parts
        and file.name not in {"package-manifest.json"}
    }
    (ROOT / "package-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if result["status"] != "PASS":
        raise AssertionError(result["status"])


if __name__ == "__main__":
    main()
