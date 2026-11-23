"""Retain failed generated records across representative families and task types."""

from pathlib import Path

import pytest

from r0.modules.replay import write_prior_diagnostics


@pytest.mark.parametrize("family", ["linear", "gp", "mlp", "scm", "tree", "interaction"])
@pytest.mark.parametrize("trace_json", ['{"latent":[NaN]}', '{"latent":'])
def test_malformed_records_leave_complete_reports_for_every_family(
    tmp_path: Path, family: str, trace_json: str
) -> None:
    import json
    from dataclasses import replace

    from r0.modules.priors import GeneratedTask, Proposal, sample_proposal

    def malformed(proposal: Proposal) -> GeneratedTask:
        return replace(sample_proposal(proposal), trace_json=trace_json)

    config = tmp_path / "config.toml"
    config.write_text(f'version=1\nfamilies=["{family}"]\nseeds=[11]\ndraws=3\n')
    output = tmp_path / "failed"
    report = write_prior_diagnostics(config, output, sampler=malformed)
    assert report["status"] == "FAIL"
    assert report["totals"]["proposed"] == 6
    assert report["totals"]["generation_failed"] == 6
    assert report["totals"]["selection_rejected"] == 0
    assert json.loads((output / "report.json").read_text())["status"] == "FAIL"
    records = [json.loads(p.read_text()) for p in (output / "proposals").glob("*.json")]
    assert len(records) == 6
    assert {r["proposal"]["task_type"] for r in records} == {"regression", "binary", "multiclass"}
    assert all(trace_json in r["returned_record_repr"] for r in records)
    assert (output / "manifest.json").is_file()
