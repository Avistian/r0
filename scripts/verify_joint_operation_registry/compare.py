"""Compare retained baseline/candidate bytes without importing the product."""

import json
import os
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
BASE = BUNDLE / "compatibility-baseline"
CURRENT = BUNDLE / "compatibility-final"


def main():
    left = json.loads((BASE / "result.json").read_text())
    right = json.loads((CURRENT / "result.json").read_text())
    if left != right:
        raise RuntimeError("baseline/candidate digest mismatch")
    chunks = sorted(p.name for p in BASE.glob("*.txt"))
    if chunks != sorted(p.name for p in CURRENT.glob("*.txt")):
        raise RuntimeError("baseline/candidate chunk list mismatch")
    for name in chunks:
        if (BASE / name).read_bytes() != (CURRENT / name).read_bytes():
            raise RuntimeError(f"baseline/candidate exact byte mismatch: {name}")
    directory = Path(os.environ["R0_REVIEW_OUTPUT"]).resolve()
    if directory.exists():
        raise RuntimeError("comparison directory must be fresh")
    directory.mkdir(parents=True)
    output = directory / "result.json"
    output.write_text(
        json.dumps(
            {
                "verdict": "PASS",
                "cases": len(left["cases"]),
                "chunks": len(chunks),
                "comparison": "exact bytes and SHA256",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"PASS: exact bytes across {len(left['cases'])} cases/{len(chunks)} chunks")


if __name__ == "__main__":
    main()
