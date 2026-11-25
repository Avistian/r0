"""Local CPU/Colab generation; Modal launches go through agent-train.yml."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from r0.modules.generation import load_request, run_guarded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    result = run_guarded(
        load_request(args.config), args.out, workers=args.workers, retry_failed=args.retry_failed
    )
    print(json.dumps(asdict(result), indent=2))
    return 1 if result.timed_out else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
