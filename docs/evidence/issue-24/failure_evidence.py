"""Regenerate public failure examples; never construct or load the sealed case."""

import json
from dataclasses import replace
from pathlib import Path

from r0.modules.challenges import evaluate_pair, public_pairs
from r0.modules.episodes import Row

pair = public_pairs()[2]
wrong = replace(pair, after=replace(pair.after, query_targets=(0.0, 0.0)))
invalid = replace(
    pair,
    after=replace(pair.after, context=(Row("c0", (0.0,)),), context_targets=(1.0,)),
)
for name, case in (("wrong-answer", wrong), ("invalid-context", invalid)):
    report = evaluate_pair(case)
    assert report["status"] == "FAIL"
    path = Path(__file__).parent / f"{name}.json"
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        assert path.read_text() == encoded, "existing failure record differs"
    else:
        with path.open("x") as handle:
            handle.write(encoded)
    print(f"{name}: expected FAIL retained at {path}")
