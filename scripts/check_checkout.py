"""Reject verification against an r0 installation outside this checkout."""

import importlib.util
import sys
from pathlib import Path


def main() -> int:
    expected = (Path(__file__).resolve().parents[1] / "src" / "r0").resolve()
    spec = importlib.util.find_spec("r0")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    locations = (
        [Path(path).resolve() for path in spec.submodule_search_locations]
        if spec and spec.submodule_search_locations is not None
        else []
    )
    print(f"Interpreter: {sys.executable}")
    print(f"r0 source: {origin}")
    print(f"Expected package: {expected}")
    if origin != expected / "__init__.py" or locations != [expected]:
        print(
            "FAIL: r0 must resolve inside this checkout. Run make setup here, or use "
            "PYTHONPATH=src when deliberately reusing an environment."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
