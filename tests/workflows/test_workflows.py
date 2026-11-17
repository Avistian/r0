"""Caller contracts; agent guards are tested in ml-factory/tests/test_workflows.py."""

import pathlib
import re

WF = pathlib.Path(".github/workflows")
AGENTS = ("implement", "review", "explore", "implement-spec", "train")


def test_agent_callers_pin_one_factory_commit() -> None:
    pins: set[str] = set()
    for name in AGENTS:
        text = (WF / f"agent-{name}.yml").read_text()
        calls = re.findall(
            rf"uses: Avistian/ml-factory/\.github/workflows/agent-{name}\.yml@([0-9a-f]{{40}})$",
            text,
            re.M,
        )
        assert len(calls) == 1, name
        pins.update(calls)
        assert "    steps:" not in text
        assert "    runs-on:" not in text
        assert "FACTORY_PAT: ${{ secrets.FACTORY_PAT }}" in text
        assert "secrets: inherit" not in text
        assert "OPENAI_API_KEY" not in text
        assert "openai/codex-action" not in text
    assert len(pins) == 1


def test_train_caller_inputs_and_secrets() -> None:
    text = (WF / "agent-train.yml").read_text()
    assert "pull_request" not in text
    for name in ("config", "run"):
        assert f"      {name}: ${{{{ inputs.{name} }}}}" in text
    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        assert f"      {name}: ${{{{ secrets.{name} }}}}" in text


def test_spec_caller_preserves_dispatch_input() -> None:
    text = (WF / "agent-implement-spec.yml").read_text()
    assert "      spec_issue: ${{ inputs.spec_issue }}" in text


def test_non_agent_workflows_stay_hosted() -> None:
    for name in ("checks", "chain-on-merge"):
        text = (WF / f"{name}.yml").read_text()
        assert "runs-on: ubuntu-latest" in text
        assert "Avistian/ml-factory" not in text


def test_workflows_never_use_pr_edit() -> None:
    for f in WF.iterdir():
        if f.suffix in {".yml", ".yaml"}:
            assert not re.search(r"\bgh\s+pr\s+edit\b", f.read_text()), f.name
