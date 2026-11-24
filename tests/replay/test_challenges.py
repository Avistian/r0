import json
from pathlib import Path

import pytest

from r0.modules.replay import write_challenge_report


def test_public_report_is_replayable_and_never_certifies_sealed_case(tmp_path: Path):
    output = tmp_path / "report.json"
    report = write_challenge_report(output)
    assert report["public_status"] == "PASS"
    assert report["ticket_status"] == "PENDING"
    assert report["sealed_status"] == "PENDING_OWNER_VERIFICATION"
    assert report["novelty"] == "UNKNOWN"
    assert len(report["pairs"]) == 3
    content = output.read_bytes()
    assert write_challenge_report(output) == report
    assert output.read_bytes() == content
    assert json.loads(content)["suite_version"] == "public-challenges-v1"
    output.write_text("{}")
    with pytest.raises(ValueError, match="different contents"):
        write_challenge_report(output)
    assert output.read_text() == "{}"
