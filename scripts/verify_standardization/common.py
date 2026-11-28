"""Independent review evidence plumbing, no product internals imported."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("REPO", "/tmp/r0-28")).resolve()
OUT = Path(os.environ["OUT"]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
PIN = "962f0ada62a4eda3e71c45fa45bab0c4f66eb8fe"
torch.set_num_threads(1)
checks = []


def enc(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value.tolist()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, default=enc, indent=2, sort_keys=True) + "\n")


def check(name, actual, expected=True):
    ok = actual == expected
    checks.append(dict(name=name, actual=actual, expected=expected, passed=bool(ok)))
    save("checks.json", checks)
    if not ok:
        raise AssertionError(name)


def close(name, actual, expected, rtol=3e-14, atol=1e-14):
    actual, expected = np.array(actual), np.array(expected)
    ok = np.allclose(actual, expected, rtol=rtol, atol=atol)
    checks.append(
        dict(name=name, actual=actual, expected=expected, rtol=rtol, atol=atol, passed=bool(ok))
    )
    save("checks.json", checks)
    if not ok:
        raise AssertionError(name)


def rejects(name, fn):
    try:
        fn()
    except (ValueError, TypeError) as error:
        checks.append(dict(name=name, passed=True, error=repr(error)))
        save("checks.json", checks)
    else:
        check(name, False)


def custody():
    modules = {}
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not name.startswith("r0") or not file:
            continue
        file = Path(file).resolve()
        rel = file.relative_to(REPO)
        raw = file.read_bytes()
        pinned = subprocess.check_output(["git", "show", f"{PIN}:{rel}"], cwd=REPO)
        modules[name] = dict(
            path=str(file),
            sha256=hashlib.sha256(raw).hexdigest(),
            pinned_sha256=hashlib.sha256(pinned).hexdigest(),
            matches=raw == pinned,
        )
    save(
        "custody.json",
        dict(
            pin=PIN,
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            status=subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
            python=platform.python_version(),
            numpy=np.__version__,
            torch=torch.__version__,
            modules=modules,
        ),
    )
    check("transitive_imports_match_pin", all(m["matches"] for m in modules.values()))
    save("summary.json", dict(passed=len(checks), failed=sum(not c["passed"] for c in checks)))
