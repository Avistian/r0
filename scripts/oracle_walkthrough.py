"""Regenerate authored cards and inspect isolation, invariance and failure evidence."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from r0.modules.episodes import Episode, Feature, Row
from r0.modules.oracles import GPSpec, gp
from r0.modules.replay import oracle_replay, parse_oracle_config, write_oracle_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    observations: list[str] = []
    for path in sorted((ROOT / "configs/oracles").glob("*.toml")):
        report = write_oracle_report(
            path,
            args.out / f"{path.stem}.json",
            ROOT / "docs/evidence/issue-22/cards" / f"{path.stem}.json",
        )
        assert report["verification"]["status"] == "PASS"
        config = parse_oracle_config(path.read_text())
        assert oracle_replay(config) == oracle_replay(config)
        if config.mode == "posterior":
            changed_targets = (
                tuple(float(y) + 1 for y in config.query_targets)
                if isinstance(config.reference, GPSpec)
                else (config.reference.classes[-1],) * len(config.query_x)
            )
            changed = oracle_replay(replace(config, query_targets=changed_targets))
            assert changed["prediction"] == report["prediction"]
            assert changed["predictor_view"] == report["predictor_view"]
            assert changed["config_fingerprint"] != report["config_fingerprint"] or (
                changed_targets == config.query_targets
            )
            reversed_context = oracle_replay(
                replace(config, context_x=config.context_x[::-1], context_y=config.context_y[::-1])
            )
            reversed_queries = oracle_replay(
                replace(
                    config, query_x=config.query_x[::-1], query_targets=config.query_targets[::-1]
                )
            )
            extended_queries = oracle_replay(
                replace(
                    config,
                    query_x=config.query_x + config.query_x[:1],
                    query_targets=config.query_targets + config.query_targets[:1],
                )
            )
            fields = (
                ("mean", "latent_variance", "observation_variance")
                if isinstance(config.reference, GPSpec)
                else ("probabilities",)
            )
            for field in fields:
                expected = report["prediction"][field]
                np.testing.assert_allclose(
                    reversed_context["prediction"][field],
                    expected,
                    atol=config.atol,
                    rtol=config.rtol,
                )
                np.testing.assert_allclose(
                    reversed_queries["prediction"][field][::-1],
                    expected,
                    atol=config.atol,
                    rtol=config.rtol,
                )
                np.testing.assert_allclose(
                    extended_queries["prediction"][field][:-1],
                    expected,
                    atol=config.atol,
                    rtol=config.rtol,
                )
            if isinstance(config.reference, GPSpec):
                covariance = np.asarray(reversed_queries["prediction"]["latent_covariance"])
                np.testing.assert_allclose(
                    covariance[::-1, ::-1],
                    report["prediction"]["latent_covariance"],
                    atol=config.atol,
                    rtol=config.rtol,
                )
        if path.stem in ("coin-undefined-auc", "multiclass-undefined-auc"):
            assert report["scoring"]["roc_auc"]["value"] is None
            assert report["scoring"]["roc_auc"]["reason"]
        if path.stem == "gp-noiseless":
            assert report["scoring"]["gaussian_negative_log_density"]["value"] is None
        observations.append(
            f"PASS {path.stem}: independent card; replay; applicable isolation/invariance"
        )
    config = parse_oracle_config((ROOT / "configs/oracles/gp-duplicates.toml").read_text())
    assert isinstance(config.reference, GPSpec)
    invalid = {
        "singular noiseless covariance": lambda: oracle_replay(
            replace(config, reference=replace(config.reference, noise_variance=0.0))
        ),
        "oversized queries": lambda: replace(config, query_x=(0.0,) * 65),
        "negative noise": lambda: GPSpec("linear", noise_variance=-1.0),
        "unknown configuration": lambda: parse_oracle_config("version=1\nunknown=true"),
        "unsupported feature": lambda: gp(
            Episode(
                "bad",
                "bad",
                "regression",
                (Feature("x", "count"),),
                (Row("c", (1,)),),
                (1.0,),
                (Row("q", (2,)),),
                (2.0,),
            ).predictor_view(),
            GPSpec("linear"),
        ),
        "missing coordinate": lambda: gp(
            Episode(
                "bad",
                "bad",
                "regression",
                (Feature("x", "continuous"),),
                (Row("c", (None,)),),
                (1.0,),
                (Row("q", (2.0,)),),
                (2.0,),
            ).predictor_view(),
            GPSpec("linear"),
        ),
    }
    for name, attempt in invalid.items():
        try:
            attempt()
        except ValueError as error:
            observations.append(f"EXPECTED ERROR {name}: {error}")
        else:
            raise AssertionError(f"invalid case accepted: {name}")
    elapsed = time.perf_counter() - started
    receipt = {
        "observations": observations,
        "elapsed_seconds": elapsed,
        "timing_status": "observed, not a hardware-independent acceptance threshold",
    }
    with (args.out / "walkthrough.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print("\n".join(observations))
    print(f"Observed walkthrough duration: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
