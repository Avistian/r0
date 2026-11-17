"""tests/workflows/test_workflows.py: compute only happens in agent-train.yml (ADR 0004)."""

import pathlib
import re

WF = pathlib.Path(".github/workflows")
FORBIDDEN = re.compile(r"modal|make train|make data|make eval\b(?!-)", re.IGNORECASE)


def test_only_the_train_workflow_mentions_compute() -> None:
    for f in WF.glob("*.yml"):
        if f.name == "agent-train.yml":
            continue
        assert not FORBIDDEN.search(f.read_text()), f"{f.name} must not start compute"


def test_train_workflow_never_runs_on_pull_requests() -> None:
    text = (WF / "agent-train.yml").read_text()
    assert "pull_request" not in text
