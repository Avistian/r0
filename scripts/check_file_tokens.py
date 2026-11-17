"""scripts/check_file_tokens.py: fails when a file is too big for an agent to edit in one go.
Usage: python scripts/check_file_tokens.py <file>..."""

import pathlib
import re
import sys

MAX_TOKENS, BYTES_PER_TOKEN = 5000, 4
IGNORED = [re.compile(p) for p in (r"(^|/)uv\.lock$", r"^\.github/workflows/[^/]+\.yml$")]

over = []
for f in sys.argv[1:]:
    p = pathlib.Path(f)
    if any(r.search(f) for r in IGNORED) or not p.is_file():
        continue
    data = p.read_bytes()
    if b"\0" in data[:8000]:
        continue  # binary files are handled by the 1 MB rule in the hook
    tokens = -(-len(data) // BYTES_PER_TOKEN)
    if tokens > MAX_TOKENS:
        over.append((f, tokens))
if over:
    print(f"These files are over {MAX_TOKENS} tokens (bytes / 4):")
    for f, t in over:
        print(f"  {f}: ~{t} tokens")
    print("Split each file before you commit. Do not delete code or squash it onto fewer lines.")
    sys.exit(1)
