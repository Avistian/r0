"""Compare retained pre/post-refactor predictions and scores, preserving both values."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rows = []
    for task in ("binary", "multiclass", "regression"):
        for name in ("simple", "linear", "boosted_tree", "learner", "tabpfn"):
            for draw in (1, 2, 3):
                relative = f"{task}-results/{name}-{draw}"
                before, after = args.before / relative, args.after / relative
                for file in ("input.json", "scoring.json"):
                    assert (before / file).read_bytes() == (after / file).read_bytes()
                values = [np.load(p / "output/predictions.npy") for p in (before, after)]
                records = [json.loads((p / "result.json").read_text()) for p in (before, after)]
                ids = [
                    json.loads((p / "output/prediction.json").read_text())["query_ids"]
                    for p in (before, after)
                ]
                assert ids[0] == ids[1]
                assert all(r["status"] == "SUCCESS" for r in records)
                np.testing.assert_allclose(values[0], values[1], rtol=0, atol=0)
                assert records[0]["scores"] == records[1]["scores"]
                rows.append(
                    {
                        "attempt": relative,
                        "query_ids": ids[0],
                        "before": values[0].tolist(),
                        "after": values[1].tolist(),
                        "before_scores": records[0]["scores"],
                        "after_scores": records[1]["scores"],
                    }
                )
    args.out.mkdir(exist_ok=True)
    for task in ("binary", "multiclass", "regression"):
        subset = [row for row in rows if row["attempt"].startswith(task + "-")]
        (args.out / f"parity-{task}.json").write_text(
            json.dumps({"status": "PASS", "evidence": "FIXTURE_ONLY", "attempts": subset}, indent=2)
            + "\n"
        )
    print(f"PASS: {len(rows)} attempts, exact predictions and scores, identical input/scoring")


if __name__ == "__main__":
    main()
