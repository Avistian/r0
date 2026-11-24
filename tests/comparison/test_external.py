"""Versioned external predictor boundary, exercised in a real local subprocess."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from r0.modules import comparison

FIXTURE = Path("docs/evidence/issue-26/fixtures/binary.json")


def test_compare_reports_fixture_gaps_and_quality(tmp_path: Path) -> None:
    comparison.prepare(FIXTURE, tmp_path / "prepared")
    report = comparison.compare(
        tmp_path / "prepared", tmp_path / "report", diagnostic=True, fixture_predictors=True
    )
    assert report["comparison"]["status"] == "COMPLETE"
    assert report["comparison"]["evidence"] == "FIXTURE_ONLY"
    assert report["comparison"]["gap"]["mean"] == 0.0
    assert report["comparison"]["quality_target_met"] is False
    assert report["groups"][0]["mean_gap"] == 0.0
    assert report["status"] == "INCOMPLETE"  # no real measurement claimed


def test_learner_backend_is_integrated_into_saved_comparison(tmp_path: Path) -> None:
    from r0.modules.comparison import environment

    comparison.prepare(FIXTURE, tmp_path / "prepared")
    runtime = environment()
    runtime["thread_environment"] = runtime["required_thread_environment"]
    identity = {
        "implementation": "authored-backend-v2",
        "environment": runtime,
        "settings": {"fixed": True},
        "checkpoint": "FIXTURE_NO_CHECKPOINT",
        "evidence": "FIXTURE_ONLY",
    }
    script = tmp_path / "adapter.py"
    script.write_text("""import argparse,json,numpy as np
from pathlib import Path
from r0.modules.comparison import prediction_worker
class Backend:
 def fit(self, context): self.context=context
 def predict(self, queries):
  assert 'targets' not in queries
  return queries['row_ids'], np.tile([0.5,0.5],(len(queries['row_ids']),1))
p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--out');p.add_argument('--config')
a=p.parse_args();prediction_worker(Path(a.input),Path(a.out),Backend(),json.loads(Path(a.config).read_text())["identity"])
""")
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps({"identity": identity}))
    descriptor = comparison.ExternalPredictor(
        "learner", Path(sys.executable), identity_path, script
    )
    paths = [tmp_path / f"attempt-{draw}" for draw in (1, 2, 3)]
    for draw, path in enumerate(paths, 1):
        assert (
            comparison.attempt(tmp_path / "prepared", path, descriptor, draw, diagnostic=True)[
                "status"
            ]
            == "SUCCESS"
        )
    reference = replace(descriptor, name="tabpfn")  # explicitly the same authored substitute
    report = comparison.compare(
        tmp_path / "prepared",
        tmp_path / "report",
        diagnostic=True,
        external_predictors=[descriptor, reference],
        saved_attempts={"learner": paths},
    )
    assert report["comparison"]["status"] == "COMPLETE"
    assert report["comparison"]["gap"]["mean"] == 0
    assert (tmp_path / "report/learner-1/input.json").read_bytes() == (
        paths[0] / "input.json"
    ).read_bytes()


@pytest.mark.parametrize("changed", ["worker", "config"])
def test_changed_execution_files_reject_saved_attempt(tmp_path: Path, changed: str) -> None:
    comparison.prepare(FIXTURE, tmp_path / "prepared")
    runtime = comparison.environment()
    runtime["thread_environment"] = runtime["required_thread_environment"]
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "identity": {
                    "implementation": "fixture",
                    "environment": runtime,
                    "settings": {"fixed": True},
                    "checkpoint": "FIXTURE_NO_CHECKPOINT",
                    "evidence": "FIXTURE_ONLY",
                }
            }
        )
    )
    script = tmp_path / "backend.py"
    script.write_text(Path("scripts/comparison_fixture_backend.py").read_text())
    descriptor = comparison.ExternalPredictor("learner", Path(sys.executable), config, script)
    result = comparison.attempt(
        tmp_path / "prepared", tmp_path / "attempt", descriptor, 1, diagnostic=True
    )
    assert result["status"] == "SUCCESS"
    path = script if changed == "worker" else config
    path.write_text(path.read_text() + "\n")
    expected = comparison.cache_identity(tmp_path / "prepared", descriptor, 1, diagnostic=True)
    with pytest.raises(ValueError, match="cache identity"):
        comparison.read_attempt(tmp_path / "attempt", expected)


def test_malformed_external_output_is_a_retained_failure(tmp_path: Path) -> None:
    comparison.prepare(FIXTURE, tmp_path / "prepared")
    identity = {
        "implementation": "malformed-fixture",
        "environment": {"fixture": True},
        "settings": {"fixed": True},
        "checkpoint": "FIXTURE_NO_CHECKPOINT",
        "evidence": "FIXTURE_ONLY",
    }
    script = tmp_path / "backend.py"
    script.write_text(
        """import argparse,json,numpy as np
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--out');p.add_argument('--config');a=p.parse_args()
v=json.loads(Path(a.input).read_text());o=Path(a.out);o.mkdir()
np.save(o/'predictions.npy',np.tile([0.5,0.5],(len(v['query_ids']),1)))
identity="""
        + repr(identity)
        + """
(o/'prediction.json').write_text(json.dumps({'version':1,'backend_identity':identity,
 'environment':identity['environment'],'invariance':'PASS','evidence':{'fit_ids':v['context_ids']}}))
"""
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"identity": identity}))
    descriptor = comparison.ExternalPredictor("learner", Path(sys.executable), config, script)
    result = comparison.attempt(
        tmp_path / "prepared", tmp_path / "attempt", descriptor, 1, diagnostic=True
    )
    assert result["status"] == "FAILED"
    assert "query" in result["reason"]
    assert json.loads((tmp_path / "attempt/result.json").read_text())["status"] == "FAILED"
    assert (tmp_path / "attempt/integrity.json").is_file()
