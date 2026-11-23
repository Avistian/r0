from collections import Counter
from pathlib import Path

import pytest

from r0.modules.replay import diagnostic_schedule, parse_diagnostic_config, write_prior_diagnostics


def test_default_schedule_balances_types_and_command_retains_selection_evidence(
    tmp_path: Path,
) -> None:
    config = parse_diagnostic_config("version=1\n")
    proposals = diagnostic_schedule(config)
    assert len(proposals) == 72
    cells = Counter((p.family, p.noise, p.task_type) for p in proposals)
    assert set(cells.values()) == {2}
    path = tmp_path / "config.toml"
    path.write_text('version=1\nfamilies=["linear"]\nselection="drop_added_noise"\n')
    report = write_prior_diagnostics(path, tmp_path / "run")
    assert report["totals"]["proposed"] == 12
    assert report["totals"]["selected"] == 6
    assert (tmp_path / "run/coverage.md").is_file()
    assert len(list((tmp_path / "run/proposals").glob("*.json"))) == 12
    path.write_text("version=1\nrows=96\n")
    with pytest.raises(ValueError, match="existing"):
        write_prior_diagnostics(path, tmp_path / "run")
    with pytest.raises(ValueError, match="limit"):
        parse_diagnostic_config("version=1\ndraws=100\n")


def test_generation_failures_are_retained_before_failure_report(tmp_path: Path) -> None:
    from r0.modules.priors import GeneratedTask, Proposal

    def broken(proposal: Proposal) -> GeneratedTask:
        raise ValueError(f"injected failure {proposal.id}")

    path = tmp_path / "config.toml"
    path.write_text('version=1\nfamilies=["linear"]\nseeds=[11]\ndraws=1\n')
    report = write_prior_diagnostics(path, tmp_path / "failed", sampler=broken)
    assert report["status"] == "FAIL"
    assert report["totals"]["generation_failed"] == 2
    assert len(list((tmp_path / "failed/proposals").glob("*.json"))) == 2
    assert (tmp_path / "failed/manifest.json").is_file()


def test_invalid_generated_record_is_retained_before_reporting_failure(tmp_path: Path) -> None:
    from dataclasses import replace

    from r0.modules.priors import GeneratedTask, Proposal, sample_proposal

    def malformed(proposal: Proposal) -> GeneratedTask:
        return replace(sample_proposal(proposal), trace_json='{"latent":[NaN]}')

    path = tmp_path / "config.toml"
    path.write_text('version=1\nfamilies=["linear"]\nseeds=[11]\ndraws=1\n')
    report = write_prior_diagnostics(path, tmp_path / "malformed", sampler=malformed)
    assert report["status"] == "FAIL"
    assert len(list((tmp_path / "malformed/proposals").glob("*.json"))) == 2
    assert (tmp_path / "malformed/report.json").is_file()
