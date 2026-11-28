import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from r0.modules import priors as p

OUT = Path(
    os.environ.get("ROOT_CONVERTER_EVIDENCE", str(Path(__file__).parent / "evidence"))
).resolve()
OUT.mkdir(parents=True, exist_ok=True)
REPO = Path(os.environ.get("ROOT_CONVERTER_REPO", "/tmp/r0-28")).resolve()
HEAD = "c997bc2dc98db7611d7aa78552ee95467093301e"
BASE = "6ba0bc2813b777749c9944d93b404ef0348168c3"
checks = []
torch.set_num_threads(1)


def encode(x):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, default=encode, indent=2, sort_keys=True) + "\n")


def custody(name):
    modules = {
        n: {
            "path": str(Path(m.__file__).resolve()),
            "sha256": hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        }
        for n, m in tuple(sys.modules.items())
        if n.startswith("r0") and getattr(m, "__file__", None) and Path(m.__file__).is_file()
    }
    save(
        name,
        dict(
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            status=subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
            python=platform.python_version(),
            numpy=np.__version__,
            torch=torch.__version__,
            imports=modules,
            scripts={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
        ),
    )


def ok(name, condition, **evidence):
    row = dict(name=name, status="PASS" if condition else "REJECT", **evidence)
    checks.append(row)
    save("checks.json", checks)
    if not condition:
        raise AssertionError(name)


def close(name, actual, expected, tol=2e-11):
    a = np.asarray(actual)
    e = np.asarray(expected)
    error = float(np.max(np.abs(a - e))) if a.size else 0
    ok(
        name,
        a.shape == e.shape and bool(np.allclose(a, e, rtol=tol, atol=tol)),
        max_abs_error=error,
        shape=list(a.shape),
    )


def reject(name, fn):
    try:
        fn()
    except (ValueError, TypeError) as exc:
        ok(name, True, rejection=str(exc))
        return
    ok(name, False, reason="accepted malformed or incompatible record")


def rng(seed, task_id, *keys):
    # Independently reconstruct documented replay identity; no internal RNG helper.
    b = json.dumps(
        ["causal-v1", seed, task_id, keys], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(b).digest(), "big"))


def draw(law, n, d, seed, task_id, node=0, root=False, cal=False):
    prefix = (("calibration",) if cal else ()) + (("root-input-v1",) if root else ())
    a = law.inputs
    cols = []
    for j in range(d):
        r = rng(seed, task_id, *prefix, "innovation", node, j)
        signs = rng(seed, task_id, *prefix, "innovation", node, j, "signs")

        def sample(k, r=r, signs=signs):
            if a.distribution == "gaussian":
                return r.standard_normal(k)
            if a.distribution == "uniform":
                return (r.random(k) * 2 - 1) * np.sqrt(3)
            if a.distribution == "student_t":
                return r.standard_t(a.df, k)
            return r.standard_normal(k) + a.mixture_gap * (2 * signs.integers(0, 2, k) - 1)

        if a.bound is None:
            v = sample(n)
        else:
            # Scalar acceptance reconstructs the accepted stream independently.
            v = []
            while len(v) < n:
                z = sample(1)[0]
                if abs(z) <= a.bound:
                    v.append(z)
            v = np.asarray(v)
        cols.append(a.location + a.scale * v)
    out = np.column_stack(cols)
    if law.geometry == "ball":
        radial = rng(seed, task_id, *prefix, "innovation", node, "radius").random(n) ** (1 / d)
        out = out / np.linalg.norm(out, axis=1)[:, None] * radial[:, None]
    if law.factor:
        out = out @ np.array(law.factor)
    return out


def task_settings(kind="regression", rows=48):
    return p.LinearRecipe(
        rows=rows,
        context_rows=19,
        features=2,
        signal_features=2,
        active_features=2,
        task_type=kind,
        classes={"regression": 0, "binary": 2, "multiclass": 3}[kind],
    )


def plain(root, rows=48):
    return p.CausalRecipe(
        task=task_settings(rows=rows),
        nodes=(root,),
        observed=((0, 0), (0, 1)),
        target=((0, 2),),
        version="causal-v2",
    )


def trace(task, indices):
    t = p.explain_generation(task, indices)
    return t, {n["id"]: np.asarray(n["values"]) for n in t["nodes"]}


def fitted(kind, C, seed, scale=1.7):
    a = p.calibrate_source_activation(
        p.CausalActivation(kind),
        C,
        rng=np.random.default_rng(seed),
        scale=scale,
        calibration_id=f"own-independent-grid-{seed}",
    )
    m = C.mean(0)
    inv = 1 / (C.std(0) + 1e-4)
    rows = np.random.default_rng(seed).integers(len(C), size=C.shape[1])
    shift = -(C[rows, np.arange(C.shape[1])] - m) * inv
    z = scale * ((C - m) * inv + shift)
    v = z * z if kind == "square" else np.tanh(z) if kind == "tanh" else np.ones_like(z)
    om = v.mean(0)
    oi = 1 / (v.std(0) + 1e-4)
    for field, expected in [
        ("input_mean", m),
        ("input_inverse_std", inv),
        ("shift_rows", rows),
        ("shift", shift),
        ("output_mean", om),
        ("output_inverse_std", oi),
    ]:
        close(f"fit-{seed}-{field}", getattr(a, field), expected)
    digest = hashlib.sha256(
        f"{C.shape[0]},{C.shape[1]}:".encode() + np.asarray(C, dtype="<f8").tobytes()
    ).hexdigest()
    ok(f"fit-{seed}-digest", a.calibration_digest == digest)

    def fun(x):
        zz = scale * ((x - m) * inv + shift)
        vv = zz * zz if kind == "square" else np.tanh(zz) if kind == "tanh" else np.ones_like(zz)
        return (vv - om) * oi

    save(
        f"fit-{seed}.json",
        dict(
            sample=C,
            record=asdict(a),
            independent=dict(
                mean=m, inverse=inv, shift_rows=rows, shift=shift, output_mean=om, output_inverse=oi
            ),
        ),
    )
    return a, fun
