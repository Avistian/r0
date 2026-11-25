"""Subprocess entrypoint used by run_guarded; configuration arrives over stdin."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from r0.modules.generation import RunRequest, run_generation
from r0.modules.priors import LinearRecipe


def main() -> int:
    data = json.load(sys.stdin)
    config = data["request"]
    recipe = config.pop("recipe")
    recipe["class_offsets"] = tuple(recipe["class_offsets"])
    request = RunRequest(LinearRecipe(**recipe), **config)
    sync_path: str | None = data["sync_path"]

    def persist() -> None:
        if sync_path is not None:
            subprocess.run(["sync", sync_path], check=True, capture_output=True)

    result = run_generation(
        request,
        Path(data["out"]),
        workers=data["workers"],
        retry_failed=data["retry_failed"],
        persist=persist,
    )
    persist()
    print(json.dumps(asdict(result), sort_keys=True))
    return 0 if result.status == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
