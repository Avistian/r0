"""Accepted-spec oracle: a display underflow must not discard a valid categorical law."""

import argparse
import hashlib
import inspect
import json
import math
import sys
from dataclasses import asdict
from decimal import Decimal, localcontext
from pathlib import Path

from r0.modules import priors as p


def run(out):
    out.mkdir(parents=True)
    request = p.CategoricalParameter("boundary", ("left", "middle", "right"))
    decay_request = p.NumericalParameter(
        "simple_random_weights_decay_rate",
        0.1 / math.log(4),
        6.0,
        shape=(1,),
        scale="log",
        boundary_mass=True,
    )
    noise_request = p.NumericalParameter(
        "simple_random_weights_std_scale",
        0.0001,
        10.0,
        shape=(1,),
        scale="log",
    )
    hyper = p.ParameterPreferences(
        preferences=tuple(
            p.NumericalPreference(r, mode="meta", mean=0.5, concentration=1.0)
            for r in (decay_request, noise_request)
        ),
        draws=(
            p.NumericalDraw(decay_request, 0, (0.0,), (decay_request.low,)),
            p.NumericalDraw(noise_request, 1, (1.0,), (10.0,)),
        ),
    )
    vector = p.PolynomialWeightVector(
        draws=(0.0, -74.495, 0.007924812503605782),
        decay=decay_request.low,
        log_noise_scale=10.0,
        permutation=(0, 1, 2),
    )
    weights = vector.values()
    assert all(math.isfinite(w) and w > 0 for w in weights)
    assert weights[0] == weights[2]
    assert weights[1] == math.ulp(0.0)
    # Integer common-denominator masses. At unit 1/2, half the positive middle
    # mass lies on either side, hence index 1 must be selected exactly.
    ratios = [w.as_integer_ratio() for w in weights]
    denominator = max(d for _, d in ratios)
    masses = [n * (denominator // d) for n, d in ratios]
    total = sum(masses)
    selected = next(i for i in range(3) if total < 2 * sum(masses[: i + 1]))
    assert selected == 1
    assert weights[1] / math.fsum(weights) == 0.0
    raw_preference = {
        "request": asdict(request),
        "mode": "meta",
        "unit": None,
        "weights": asdict(vector),
        "hyperparameters": asdict(hyper),
    }
    raw = {
        "preferences": [raw_preference],
        "draws": [asdict(p.CategoricalDraw(request, 0, 0.5, 1, "middle"))],
        "default_mode": "meta",
        "policy": "shared",
        "version": "categorical-preferences-v1",
        "limits": asdict(p.PreferenceLimits()),
    }
    (out / "input.json").write_text(json.dumps(raw, indent=2) + "\n")
    outcomes = []
    for name, construct in (
        (
            "public_constructor",
            lambda: p.CategoricalPreference(request, "meta", weights=vector, hyperparameters=hyper),
        ),
        (
            "public_decoder",
            lambda: p.categorical_preferences_from_dict(json.loads(json.dumps(raw))),
        ),
    ):
        try:
            record = construct()
        except ValueError as exc:
            outcomes.append(
                {
                    "probe": name,
                    "status": "REJECT",
                    "actual": "valid law rejected",
                    "error": str(exc),
                    "expected": "retain law; exact unit0.5 selects middle",
                }
            )
        else:
            pref = record if name == "public_constructor" else record.preferences[0]
            actual = pref.choose(0.5)
            with localcontext() as context:
                context.prec = 100
                exact_total = sum(Decimal(w) for w in weights)
                expected_logs = tuple(float((Decimal(w) / exact_total).ln()) for w in weights)
            logs = pref.log_probabilities
            assert all(math.isfinite(v) for v in logs)
            assert all(
                math.isclose(a, b, abs_tol=2e-13) for a, b in zip(logs, expected_logs, strict=True)
            )
            try:
                _ = pref.probabilities
            except ValueError as exc:
                view_error = str(exc)
            else:
                raise AssertionError("unrepresentable float probability view was silently clipped")
            assert "underflow" in view_error
            assert pref.choose(0.5) == selected
            outcomes.append(
                {
                    "probe": name,
                    "status": "PASS" if actual == selected else "REJECT",
                    "actual": actual,
                    "log_probabilities": logs,
                    "independent_decimal_logs": expected_logs,
                    "probability_view_error": view_error,
                    "expected": selected,
                }
            )
    source = Path(inspect.getfile(p.CategoricalPreference))
    result = {
        "status": "REJECT" if any(r["status"] == "REJECT" for r in outcomes) else "PASS",
        "weights": weights,
        "independent_exact_index": selected,
        "masses_numerator_strings": [str(v) for v in masses],
        "common_denominator_string": str(denominator),
        "unit": 0.5,
        "display_probability": 0.0,
        "outcomes": outcomes,
        "python": sys.executable,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "outcomes": outcomes}))
    return 1 if result["status"] == "REJECT" else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    raise SystemExit(run(parser.parse_args().output))
