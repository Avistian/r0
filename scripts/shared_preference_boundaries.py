"""Authored public categorical records at exact weighted boundaries and underflow."""

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from r0.modules import priors as p


def run(out):
    out.mkdir(parents=True)
    sampler = p.CategoricalSampler(np.random.default_rng(2821))
    sampler.categorical(p.CategoricalParameter("boundary", ("left", "middle", "right")))
    pref = sampler.freeze().preferences[0]
    hyper = pref.hyperparameters
    # Both endpoints are explicit allowed meta-Beta realizations; the small decay
    # makes the centered binary64 weights equal after rounding.
    low = hyper.draws[0].request.low
    draws = (
        replace(hyper.draws[0], units=(0.0,), values=(low,)),
        replace(hyper.draws[1], units=(1.0,), values=(10.0,)),
    )
    hyper = replace(hyper, draws=draws)
    outcomes = []
    for name, scores in (
        ("exact-half", (0.0, -1000.0, 0.0)),
        ("positive-probability-underflow", (0.0, -745.0, 0.0)),
    ):
        vector = replace(
            pref.weights,
            decay=low,
            log_noise_scale=10.0,
            draws=tuple((s + low * math.log(i + 1)) / 10.0 for i, s in enumerate(scores)),
            permutation=(0, 1, 2),
        )
        raw = {
            "weights": asdict(vector),
            "hyperparameters": asdict(hyper),
            "binary_weights": vector.values(),
        }
        (out / (name + ".json")).write_text(json.dumps(raw, indent=2) + "\n")
        if name == "exact-half":
            authored = replace(pref, weights=vector, hyperparameters=hyper)
            assert vector.values()[0] == vector.values()[2]
            assert vector.values()[1] == 0.0
            units = (
                (math.nextafter(0.5, 0.0), 0),
                (0.5, 2),
                (math.nextafter(0.5, 1.0), 2),
                (0.0, 0),
            )
            for unit, expected in units:
                assert authored.choose(unit) == expected
                outcomes.append(
                    {
                        "case": name,
                        "unit": unit,
                        "expected": expected,
                        "actual": authored.choose(unit),
                        "status": "PASS",
                    }
                )
        else:
            assert vector.values()[1] > 0.0
            assert vector.values()[1] / math.fsum(vector.values()) == 0.0
            authored = replace(pref, weights=vector, hyperparameters=hyper)
            assert authored.choose(0.5) == 1
            assert all(math.isfinite(v) for v in authored.log_probabilities)
            try:
                _ = authored.probabilities
            except ValueError as exc:
                view_error = str(exc)
            else:
                raise AssertionError("unrepresentable float probability view was silently clipped")
            assert "underflow" in view_error
            outcomes.append(
                {
                    "case": name,
                    "status": "PASS",
                    "actual": authored.choose(0.5),
                    "expected": 1,
                    "probability_view_error": view_error,
                    "log_probabilities": authored.log_probabilities,
                }
            )
        if name == "exact-half":
            assert authored.log_probabilities[1] == -math.inf
            assert authored.log_probabilities[0] == authored.log_probabilities[2]
            outcomes.append(
                {
                    "case": "zero-weight-log-view",
                    "status": "PASS",
                    "log_probabilities": [
                        "-inf" if v == -math.inf else v for v in authored.log_probabilities
                    ],
                }
            )
    for mode in ("local", "global"):
        uniform = p.CategoricalPreference(
            pref.request, mode, unit=0.5 if mode == "global" else None
        )
        assert uniform.log_probabilities == (-math.log(3),) * 3
        outcomes.append(
            {
                "case": mode + "-uniform-log-view",
                "status": "PASS",
                "log_probabilities": uniform.log_probabilities,
            }
        )
    (out / "outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    (out / "summary.json").write_text(
        json.dumps({"status": "PASS", "probes": len(outcomes)}) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
