"""Authored mean/frequency backend for subprocess verification; never a trained model."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.comparison import adapter_worker


class FixtureBackend:
    def __init__(self) -> None:
        self.context: dict[str, Any] = {}

    def fit(self, context: dict[str, Any]) -> None:
        self.context = context

    def predict(self, queries: dict[str, Any]) -> tuple[list[str], NDArray[np.float64]]:
        if "targets" in queries:
            raise ValueError("query targets reached predictor")
        context = self.context
        n = len(queries["row_ids"])
        targets = context["targets"]
        if context["task"] == "regression":
            values = np.full(n, sum(targets) / len(targets), dtype=np.float64)
        else:
            values = np.tile([targets.count(c) / len(targets) for c in context["classes"]], (n, 1))
        return queries["row_ids"], values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()
    adapter_worker(args.input, args.out, FixtureBackend(), json.loads(args.identity.read_text()))


if __name__ == "__main__":
    main()
