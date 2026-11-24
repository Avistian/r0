"""Immutable public challenge reports. No sealed case or private receipt is loaded."""

import json
import os
import platform
import tempfile
from pathlib import Path
from typing import TypedDict

import r0.modules.challenges as challenges
from r0.modules.challenges import PairReport, evaluate_pair, public_pairs
from r0.modules.replay.lib.identity import fingerprint


class ChallengeReport(TypedDict):
    suite_version: str
    public_status: str
    ticket_status: str
    sealed_status: str
    novelty: str
    training_mixture_identity: None
    influenced_development: bool
    pairs: list[PairReport]
    environment: dict[str, str]
    numerical_policy: dict[str, float]
    limits: dict[str, int]
    provenance: dict[str, str]
    content_fingerprint: str


def write_challenge_report(output_path: Path) -> ChallengeReport:
    """Execute exactly three authored pairs and retain their comparisons without overwrite."""
    pairs = [evaluate_pair(pair) for pair in public_pairs()]
    module = Path(challenges.__file__).parent
    source = {str(p.relative_to(module)): p.read_text() for p in sorted(module.rglob("*.py"))}
    report: ChallengeReport = {
        "suite_version": "public-challenges-v1",
        "public_status": "PASS" if all(p["status"] == "PASS" for p in pairs) else "FAIL",
        "ticket_status": "PENDING",
        "sealed_status": "PENDING_OWNER_VERIFICATION",
        "novelty": "UNKNOWN",
        "training_mixture_identity": None,
        "influenced_development": True,
        "pairs": pairs,
        "environment": {"python": platform.python_version()},
        "numerical_policy": {"atol": 1e-10, "rtol": 1e-10},
        "limits": {"context_rows": 12, "query_rows": 6, "features": 4, "command_seconds": 300},
        "provenance": {
            "construction": "independently authored literals in challenges/lib/cases.py",
            "answers": "authored targets and calculations, independent of reference predictions",
            "challenge_source_fingerprint": fingerprint(source),
            "replay_source_fingerprint": fingerprint(Path(__file__).read_text()),
            "preprocessing": "raw coordinates and original target units; no fitted preprocessing",
        },
        "content_fingerprint": "",
    }
    report["content_fingerprint"] = fingerprint(report)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output_path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
        try:
            os.link(temporary, output_path)
        except FileExistsError:
            if output_path.read_text() != encoded:
                raise ValueError(
                    "output already has different contents; choose a new path"
                ) from None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return report
