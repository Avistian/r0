"""Retained independent verifier support; all public r0 imports are fingerprinted."""

import atexit
import hashlib
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / Path(sys.argv[0]).stem
OUT.mkdir(exist_ok=True)
CHECKS = []


def encode(value):
    def convert(item):
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, (np.ndarray, torch.Tensor)):
            return item.tolist()
        raise TypeError(type(item).__name__)

    return json.dumps(value, sort_keys=True, default=convert, allow_nan=False)


def check(label, condition, detail=None):
    record = {"label": label, "pass": bool(condition), "detail": detail}
    CHECKS.append(record)
    with (OUT / "checks.jsonl").open("a") as stream:
        stream.write(encode(record) + "\n")
    assert condition, label


def reject(label, function):
    try:
        function()
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        check(label, True, {"rejected": type(error).__name__, "message": str(error)})
    else:
        check(label, False, "accepted adversary")


def raw(value):
    return json.loads(encode(value))


def save(name, value):
    (OUT / (name + ".json")).write_text(encode(value))


def finish():
    imports = {}
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("r0") and filename and Path(filename).is_file():
            imports[name] = {
                "path": filename,
                "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
            }
    save("imports", imports)
    save(
        "environment",
        {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "source": "be3643989df7349a295a62e242db9522a03cda5d",
        },
    )
    save("totals", {"checks": len(CHECKS), "passed": sum(c["pass"] for c in CHECKS)})


atexit.register(finish)
