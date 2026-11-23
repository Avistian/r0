"""Enforce one text-size policy on Git index blobs in Make, hooks and CI."""

import os
import re
import subprocess
import sys

MAX_TOKENS, BYTES_PER_TOKEN = 5000, 4
IGNORED = [re.compile(p) for p in (r"(^|/)uv\.lock$", r"^\.github/workflows/[^/]+\.yml$")]


def main() -> int:
    if len(sys.argv) != 1:
        print("Usage: python scripts/check_file_tokens.py (checks all Git index files)")
        return 2
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"]).split(b"\0")
    blobs: list[tuple[str, str]] = []
    for entry in entries:
        if not entry:
            continue
        metadata, encoded_path = entry.split(b"\t", 1)
        mode, identity, _stage = metadata.split()
        path = os.fsdecode(encoded_path)
        if mode != b"160000" and not any(pattern.search(path) for pattern in IGNORED):
            blobs.append((identity.decode("ascii"), path))
    if not blobs:
        return 0
    sizes = subprocess.check_output(
        ["git", "cat-file", "--batch-check"],
        input="".join(f"{identity}\n" for identity, _ in blobs),
        text=True,
    ).splitlines()
    oversized: list[tuple[str, int]] = []
    for (identity, path), row in zip(blobs, sizes, strict=True):
        _identity, kind, size = row.split()
        if kind != "blob":
            raise ValueError(f"unexpected index object type: {kind}")
        tokens = -(-int(size) // BYTES_PER_TOKEN)
        if tokens <= MAX_TOKENS:
            continue
        content = subprocess.check_output(["git", "cat-file", "blob", identity])
        if b"\0" not in content[:8000]:
            oversized.append((path, tokens))
    if oversized:
        print(f"These Git index files are over {MAX_TOKENS} tokens (bytes / 4):")
        for path, tokens in oversized:
            print(f"  {path!r}: ~{tokens} tokens")
        print("Split each file before staging. Do not delete code or squash it onto fewer lines.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
