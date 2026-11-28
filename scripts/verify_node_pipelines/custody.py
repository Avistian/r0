"""Record exact verifier scripts and loaded repository module bytes."""

import hashlib
import json
import platform
import sys
from pathlib import Path


def record_custody(destination):
    scripts = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    imports = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename and name.startswith("r0") and Path(filename).is_file():
            imports[name] = dict(
                path=filename, sha256=hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            )
    destination.write_text(
        json.dumps(
            dict(python=platform.python_version(), scripts=scripts, imports=imports), indent=2
        )
        + "\n"
    )
