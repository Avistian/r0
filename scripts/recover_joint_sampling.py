"""Restore retained joint-sampling evidence and verify its original manifest."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recover(metadata: Path, output: Path, repo: Path) -> dict[str, object]:
    meta = json.loads(metadata.read_text())
    if "parts" in meta:
        chunks = []
        for item in meta["parts"]:
            chunk = (metadata.parent / item["path"]).read_bytes()
            assert len(chunk) == item["bytes"]
            assert sha(chunk) == item["sha256"]
            chunks.append(chunk)
        data = b"".join(chunks)
    else:
        data = (metadata.parent / meta["archive"]).read_bytes()
    assert len(data) == meta["bytes"]
    assert sha(data) == meta["sha256"]
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(output)
    duplicates = output / "duplicate-paths.json"
    if duplicates.exists():
        for target, source in json.loads(duplicates.read_text()).items():
            path = output / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((output / source).read_bytes())
    for item in json.loads((output / "RECONSTRUCTION.json").read_text()):
        command = item["command"]
        assert command[:2] == ["git", "archive"]
        raw = subprocess.check_output(command, cwd=repo)
        assert len(raw) == item["bytes"]
        assert sha(raw) == item["sha256"]
        (output / item["archive"]).write_bytes(raw)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            archive.extractall(output / item["directory"], filter="data")
    manifest = (output / "MANIFEST.sha256").read_bytes()
    assert sha(manifest) == meta["manifest_sha256"]
    count = 0
    for line in manifest.decode().splitlines():
        digest, name = line.split("  ", 1)
        assert sha((output / name).read_bytes()) == digest, name
        count += 1
    assert count == meta["manifest_entries"]
    return dict(
        verdict="RESTORED_AND_VERIFIED",
        metadata=str(metadata.resolve()),
        output=str(output.resolve()),
        source=meta["source"],
        manifest_entries=count,
        archive_sha256=meta["sha256"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path, help="new directory; must not exist")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(recover(args.metadata, args.output, args.repo), indent=2))
