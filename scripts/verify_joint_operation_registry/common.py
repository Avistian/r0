"""Independent review gates and artifact handling."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

from r0.modules import priors

BUNDLE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("R0_REVIEW_ROOT", Path.cwd())).resolve()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def start():
    output = Path(os.environ["R0_REVIEW_OUTPUT"]).resolve()
    require(not output.exists(), "output directory must be fresh")
    output.mkdir(parents=True)
    pins = json.loads((BUNDLE / "pins.json").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(head in pins, "unrecognized HEAD")
    tree = subprocess.check_output(["git", "rev-parse", "HEAD:src"], cwd=ROOT, text=True).strip()
    require(tree == pins[head], "wrong source tree")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "src", "tests"], cwd=ROOT, check=True
    )
    require(
        Path(priors.__file__).resolve() == ROOT / "src/r0/modules/priors/__init__.py",
        "wrong imported checkout",
    )
    save(
        output,
        "execution",
        {
            "head": head,
            "src_tree": tree,
            "root": str(ROOT),
            "import": str(priors.__file__),
            "program_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in BUNDLE.glob("*.py")
            },
        },
    )
    return output


def save(output, name, data):
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    require(len(text.encode()) <= 20000, f"artifact too large: {name}")
    (output / f"{name}.json").write_text(text)


def rejected(action):
    try:
        action()
    except ValueError:
        return
    raise RuntimeError("malformed input unexpectedly accepted")
