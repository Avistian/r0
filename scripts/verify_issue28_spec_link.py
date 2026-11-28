"""Read-only reproduction of the retained #28 approved-specification handoff."""

import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path


def verify() -> None:
    root = Path(__file__).resolve().parents[1]
    receipt_path = root / "docs/evidence/issue-28/legacy-research/0028-spec-link-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    repository, commit, path = (receipt["repository"], receipt["spec_commit"], receipt["spec_path"])
    published = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{repository}/contents/{path}?ref={commit}"], text=True
        )
    )
    source = base64.b64decode(published["content"])
    expected = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
    assert source == expected, "published specification differs from pinned local object"
    assert hashlib.sha256(source).hexdigest() == receipt["observed_spec_sha256"]
    content = source.decode()
    decisions = re.findall(r"(?m)^(\d+)\. ", content)
    assert decisions == [str(i) for i in range(1, 20)], decisions
    assert "make sure that it covers all options from papers" in " ".join(content.split())
    link = f"https://github.com/{repository}/blob/{commit}/{path}"
    suffix = (
        "\n\n## Approved specification\n\n"
        f"[Authoritative accepted scope for #28]({link}). "
        "Implementation and verification must follow this specification.\n"
    )
    assert receipt["after_body"] == receipt["before_body"] + suffix
    current = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{repository}/issues/{receipt['issue']}"], text=True
        )
    )
    assert current["body"] == receipt["after_body"], "issue changed since retained handoff"
    print(
        json.dumps(
            {
                "spec_commit": commit,
                "spec_sha256": receipt["observed_spec_sha256"],
                "decisions": len(decisions),
                "original_body_preserved": True,
                "current_readback": "MATCH",
                "issue_state": current["state"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    verify()
