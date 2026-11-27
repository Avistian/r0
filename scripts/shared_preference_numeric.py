"""Independent public-contract checks; stop and retain the first unexpected result."""

import argparse
import copy
import hashlib
import inspect
import json
import math
import sys
import traceback
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from r0.modules import priors as p


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def count(value):
    if isinstance(value, dict):
        return sum(count(v) for k, v in value.items() if k not in ("limits", "exact_scores_limit"))
    if isinstance(value, (tuple, list)):
        return sum(map(count, value))
    return int(type(value) in (int, float))


def choose(weights, unit):
    # Integer cross-products over exact binary ratios: no production inverse-CDF helpers.
    ratios = [w.as_integer_ratio() for w in weights]
    denominator = max(d for _, d in ratios)
    masses = [n * (denominator // d) for n, d in ratios]
    un, ud = unit.as_integer_ratio()
    target, cumulative = un * sum(masses), 0
    for i, mass in enumerate(masses):
        cumulative += mass
        if target < cumulative * ud:
            return i
    raise AssertionError("oracle has no positive support")


def vector_oracle(vector):
    # Independent high-precision evaluation of polynomial weights and RMS normalization.
    with localcontext() as ctx:
        ctx.prec = 100
        raw = [
            (
                (Decimal.from_float(vector.log_noise_scale) * Decimal.from_float(z))
                - Decimal.from_float(vector.decay) * Decimal.from_float(math.log(i + 1))
            ).exp()
            for i, z in enumerate(vector.draws)
        ]
        norm = (sum(w * w for w in raw) / len(raw)).sqrt()
        return [float(raw[i] / norm) for i in (vector.permutation or range(len(raw)))]


def reject(out, key, raw, fn):
    dump(out / (key + "-input.json"), raw)
    try:
        fn(raw)
    except (ValueError, TypeError, KeyError) as exc:
        return {"key": key, "status": "REJECTED_AS_REQUIRED", "error": str(exc)}
    raise AssertionError(f"invalid record accepted: {key}")


def run(out):
    out.mkdir(parents=True)
    outcomes = []

    def record(key, **values):
        outcomes.append({"key": key, "status": "PASS", **values})
        dump(out / "outcomes.json", outcomes)

    try:
        source = Path(inspect.getfile(p.CategoricalSampler))
        dump(
            out / "identity.json",
            {
                "python": sys.executable,
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            },
        )
        request = p.CategoricalParameter("matrix", ("a", "b", "c"))
        for mode, policy in (
            ("local", "shared"),
            ("global", "shared"),
            ("meta", "shared"),
            ("meta", "refresh"),
        ):
            rng = np.random.default_rng(2817)
            sampler = p.CategoricalSampler(rng, default_mode=mode, policy=policy)
            draws = [sampler.categorical(request) for _ in range(3)]
            saved = sampler.freeze()
            raw = asdict(saved)
            key = mode + "-" + policy
            dump(out / (key + ".json"), raw)
            assert p.categorical_preferences_from_dict(json.loads(json.dumps(raw))) == saved
            expected_scopes = 3 if mode == "meta" and policy == "refresh" else 1
            assert len(saved.preferences) == expected_scopes
            assert saved.parameter_count == count(raw["preferences"]) + count(raw["draws"])
            for pref in saved.preferences:
                if mode == "meta":
                    assert pref.hyperparameters.default_mode == "meta"
                    assert len(pref.hyperparameters.draws) == 2
                    for d, law in zip(
                        pref.hyperparameters.draws, pref.hyperparameters.preferences, strict=True
                    ):
                        assert d.request.shape == (1,)
                        assert law.mode == "meta"
                        assert 0 < law.mean < 1
                        assert 0.1 <= law.concentration <= 10000
                        u = d.units[0]
                        u = min(1, max(0, 1.2 * u - 0.1)) if d.request.boundary_mass else u
                        expected = math.exp(
                            (1 - u) * math.log(d.request.low) + u * math.log(d.request.high)
                        )
                        assert math.isclose(d.values[0], expected, rel_tol=1e-14)
                    assert np.allclose(
                        pref.weights.values(), vector_oracle(pref.weights), rtol=2e-13, atol=0
                    )
                weights = pref.weights.values() if mode == "meta" else (1.0,) * 3
                for d in draws:
                    if saved.preferences[d.preference] == pref:
                        assert d.index == choose(weights, d.unit)
            assert (len({d.unit for d in draws}) == 1) == (mode == "global")
            record(key, parameters=saved.parameter_count, scopes=expected_scopes)
            # Same name in another effective mode is a distinct scope.
            explicit = sampler.categorical(
                replace(request, mode="global" if mode != "global" else "local")
            )
            assert explicit.preference == expected_scopes
            state = copy.deepcopy(rng.bit_generator.state)
            changed = replace(request, choices=("c", "b", "a"))
            outcomes.append(
                reject(
                    out,
                    key + "-contract",
                    asdict(changed),
                    lambda _, sampler=sampler, changed=changed: sampler.categorical(changed),
                )
            )
            assert rng.bit_generator.state == state

        # Authored Gaussian innovations exercise zero bins, subnormal tails, and exact boundaries.
        sampler = p.CategoricalSampler(np.random.default_rng(2821))
        sampler.categorical(request)
        pref = sampler.freeze().preferences[0]
        for key, scores, permutation in (
            ("equal-boundary", (0.0, 0.0, 0.0), (0, 1, 2)),
            ("zero-middle", (0.0, -1000.0, 0.0), (0, 1, 2)),
            ("tiny-middle", (0.0, -700.0, 0.0), (0, 1, 2)),
            ("zero-leading", (-1000.0, 0.0, 0.0), (0, 1, 2)),
            ("permuted-tail", (0.0, -700.0, -1000.0), (2, 0, 1)),
        ):
            z = tuple(
                (s + pref.weights.decay * math.log(i + 1)) / pref.weights.log_noise_scale
                for i, s in enumerate(scores)
            )
            weights = replace(pref.weights, draws=z, permutation=permutation)
            authored = replace(pref, weights=weights)
            binary = weights.values()
            assert np.allclose(binary, vector_oracle(weights), rtol=2e-12, atol=0)
            units = {
                0.0,
                math.nextafter(1.0, 0.0),
                0.5,
                math.nextafter(0.5, 0.0),
                math.nextafter(0.5, 1.0),
            }
            for j in range(1, len(binary)):
                cdf = math.fsum(binary[:j]) / math.fsum(binary)
                if cdf < 1:
                    units.update((cdf, math.nextafter(cdf, 0.0), math.nextafter(cdf, 1.0)))
            details = []
            for unit in sorted(u for u in units if 0 <= u < 1):
                expected = choose(binary, unit)
                actual = authored.choose(unit)
                assert actual == expected
                assert binary[actual] > 0
                details.append({"unit": unit, "expected": expected, "actual": actual})
            dump(
                out / (key + ".json"),
                {
                    "preference": asdict(authored),
                    "weights": binary,
                    "independent_weights": vector_oracle(weights),
                    "checks": details,
                },
            )
            record(key, checks=len(details), zero_bins=sum(w == 0 for w in binary))

        # 33 parameters includes two singleton Beta requests, vector and category draw.
        for field, limit in (("max_draws", 2), ("max_values", 2), ("max_parameters", 32)):
            rng = np.random.default_rng(28)
            sampler = p.CategoricalSampler(rng, limits=p.PreferenceLimits(**{field: limit}))
            state = copy.deepcopy(rng.bit_generator.state)
            outcomes.append(
                reject(
                    out,
                    "admit-" + field,
                    {field: limit},
                    lambda _, sampler=sampler: sampler.categorical(request),
                )
            )
            assert rng.bit_generator.state == state
            assert not sampler.freeze().draws
        for policy, expected in (("shared", 36), ("refresh", 66)):
            sampler = p.CategoricalSampler(
                np.random.default_rng(28),
                policy=policy,
                limits=p.PreferenceLimits(max_parameters=expected),
            )
            sampler.categorical(request)
            sampler.categorical(request)
            saved = sampler.freeze()
            assert saved.parameter_count == expected
            raw = json.loads(json.dumps(asdict(saved)))
            assert p.categorical_preferences_from_dict(raw) == saved
            invalid = copy.deepcopy(raw)
            invalid["limits"]["max_parameters"] -= 1
            outcomes.append(
                reject(out, policy + "-raw-budget", invalid, p.categorical_preferences_from_dict)
            )
            if policy == "refresh":
                invalid = copy.deepcopy(raw)
                invalid["policy"] = "shared"
                outcomes.append(
                    reject(out, "refresh-relabel", invalid, p.categorical_preferences_from_dict)
                )
            for key in ("beta", "weight", "orphan", "realization", "reference", "shape"):
                invalid = copy.deepcopy(raw)
                if key == "beta":
                    invalid["preferences"][0]["hyperparameters"]["preferences"][0]["mean"] = 1.0
                elif key == "weight":
                    invalid["preferences"][0]["weights"]["decay"] *= 2
                elif key == "orphan":
                    invalid["draws"] = []
                elif key == "realization":
                    d = invalid["draws"][0]
                    d["index"] = (d["index"] + 1) % 3
                    d["value"] = request.choices[d["index"]]
                elif key == "reference":
                    invalid["draws"][0]["preference"] = 20
                else:
                    invalid["preferences"][0]["hyperparameters"]["draws"][0]["request"]["shape"] = [
                        2
                    ]
                outcomes.append(
                    reject(
                        out, policy + "-bad-" + key, invalid, p.categorical_preferences_from_dict
                    )
                )
            record(policy + "-admission-and-malformed", parameters=expected)
        dump(out / "outcomes.json", outcomes)
        dump(out / "summary.json", {"status": "PASS", "probes": len(outcomes)})
    except Exception as exc:
        dump(
            out / "failure.json",
            {"status": "REJECT", "error": repr(exc), "traceback": traceback.format_exc()},
        )
        dump(out / "outcomes.json", outcomes)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
