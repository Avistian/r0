"""scripts/check_module_tests.py: fails when a module in src/r0/modules/ has no tests.
Tests go through the module's entry point; see src/r0/modules/README.md."""

import pathlib
import sys

MODULES, TESTS = pathlib.Path("src/r0/modules"), pathlib.Path("tests")
untested = [
    d.name
    for d in MODULES.iterdir()
    if d.is_dir() and not d.name.startswith("_") and not any((TESTS / d.name).glob("test_*.py"))
]
if untested:
    print("These modules have no tests:\n" + "\n".join(f"  tests/{n}/" for n in untested))
    print("\nAdd at least one test_*.py that imports the module's entry point.")
    sys.exit(1)
