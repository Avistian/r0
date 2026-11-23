"""Enforce the 1 MiB artifact limit on Git index blobs in commits and CI."""

import os
import subprocess
import sys

MAX_BYTES = 1048576


def main() -> int:
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"]).split(b"\0")
    blobs: list[tuple[str, str]] = []
    for entry in entries:
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, identity, _stage = metadata.split()
        if mode != b"160000":  # Submodules store commit identities, not file contents.
            blobs.append((identity.decode("ascii"), os.fsdecode(path)))
    if not blobs:
        return 0
    metadata = subprocess.check_output(
        ["git", "cat-file", "--batch-check"],
        input="".join(f"{identity}\n" for identity, _ in blobs),
        text=True,
    ).splitlines()
    oversized: list[tuple[str, int]] = []
    for (_, path), row in zip(blobs, metadata, strict=True):
        _identity, kind, size = row.split()
        if kind != "blob":
            raise ValueError(f"unexpected index object type: {kind}")
        if int(size) > MAX_BYTES:
            oversized.append((path, int(size)))
    if oversized:
        print("FAIL: Git index files exceed 1 MiB; data/checkpoints belong in .cache/ or Releases:")
        for path, size in oversized:
            print(f"  {path!r}: {size} bytes")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
