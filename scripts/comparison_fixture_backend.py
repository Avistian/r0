"""Authored mean/frequency backend for subprocess verification; never a trained model."""

import argparse
import json
from pathlib import Path

from r0.modules.comparison import prediction_worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prediction_worker(
        args.input, args.out, "simple", json.loads(args.config.read_text())["identity"]
    )


if __name__ == "__main__":
    main()
