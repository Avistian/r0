from pathlib import Path

import pytest

from r0.modules.replay import oracle_replay, parse_oracle_config

COIN = """
version = 1
name = "coin"
mode = "posterior"
[reference]
family = "beta_bernoulli"
classes = ["H", "T"]
concentration = [1.0, 1.0]
positive_class = "H"
[observations]
context_x = [0.0, 0.0, 0.0, 0.0]
context_y = ["H", "H", "H", "T"]
query_x = [0.0, 0.0]
[scoring]
targets = ["H", "T"]
"""


def test_oracle_replay_scores_only_queries_and_records_undefined_auc() -> None:
    report = oracle_replay(parse_oracle_config(COIN))
    scoring = report["scoring"]
    assert isinstance(scoring, dict)
    assert scoring["brier"] == pytest.approx(5 / 9)
    assert scoring["roc_auc"] == {"value": 0.5, "reason": None}
    changed = oracle_replay(
        parse_oracle_config(COIN.replace('targets = ["H", "T"]', 'targets = ["H", "H"]'))
    )
    assert changed["prediction"] == report["prediction"]
    assert changed["predictor_view"] == report["predictor_view"]
    assert changed["config_fingerprint"] != report["config_fingerprint"]
    assert isinstance(changed["scoring"], dict)
    assert changed["scoring"]["roc_auc"] == {
        "value": None,
        "reason": "requires positive and negative query targets",
    }


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (COIN.replace("version = 1", "version = true"), "version"),
        (COIN.replace("version = 1", "version = 1\nunknown = 1"), "unknown"),
        (COIN.replace("[1.0, 1.0]", "[1.0, 0.0]"), "positive"),
        (COIN.replace("query_x = [0.0, 0.0]", "query_x = [" + ",".join(["0.0"] * 65) + "]"), "64"),
        (COIN.replace('mode = "posterior"', 'mode = "prior"'), "prior"),
        (COIN + "\n[numerics]\natol = 0.01\n", "justification"),
    ],
)
def test_strict_reference_configuration(bad: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_oracle_config(bad)


def test_prior_rejects_jitter_without_conditioning_observations() -> None:
    with pytest.raises(ValueError, match="prior.*jitter"):
        parse_oracle_config(
            'version=1\nname="bad"\nmode="prior"\n'
            '[reference]\nfamily="linear"\njitter=1.0\n'
            "[observations]\nquery_x=[1.0]\n"
            '[units]\nfeature="unitless"\ntarget="unitless"\n'
        )


def test_independent_reference_comparison_reports_mismatch() -> None:
    from r0.modules.replay import compare_oracle_report

    report = oracle_replay(parse_oracle_config(COIN))
    expected = {"prediction.probabilities.0.0": 2 / 3, "scoring.roc_auc.value": 0.5}
    comparison = compare_oracle_report(report, expected, atol=1e-10, rtol=1e-10)
    assert comparison["status"] == "PASS"
    bad = compare_oracle_report(
        report, {"prediction.probabilities.0.0": 0.75}, atol=1e-10, rtol=1e-10
    )
    assert bad["status"] == "FAIL"
    with pytest.raises(ValueError, match="reference path"):
        compare_oracle_report(report, {"typo": 1.0}, atol=1e-10, rtol=1e-10)


def test_auc_orientation_multiclass_absence_and_zero_variance_density() -> None:
    from dataclasses import replace

    from r0.modules.episodes import Episode, Feature, Row
    from r0.modules.oracles import ClassificationPrediction
    from r0.modules.replay import oracle_scores

    episode = Episode(
        "auc",
        "auc",
        "binary",
        (Feature("unused", "continuous"),),
        (Row("c", (0.0,)),),
        ("H",),
        (Row("q0", (0.0,)), Row("q1", (0.0,))),
        ("H", "T"),
        ("H", "T"),
        "declared",
    )
    # Positive class is T, not the first column. A correct rank scores 1; reverse scores 0.
    prediction = ClassificationPrediction(("H", "T"), (1.0, 1.0), ((0.9, 0.1), (0.1, 0.9)), "T")
    assert oracle_scores(episode, prediction)["roc_auc"] == {"value": 1.0, "reason": None}
    assert oracle_scores(
        episode, replace(prediction, probabilities=prediction.probabilities[::-1])
    )["roc_auc"] == {"value": 0.0, "reason": None}
    root = Path(__file__).resolve().parents[2]
    for name, metric in (
        ("multiclass-undefined-auc", "roc_auc"),
        ("gp-noiseless", "gaussian_negative_log_density"),
    ):
        config = parse_oracle_config((root / f"configs/oracles/{name}.toml").read_text())
        report = oracle_replay(config)
        scoring = report["scoring"]
        assert isinstance(scoring, dict)
        assert scoring[metric]["value"] is None
        assert scoring[metric]["reason"]


def test_extreme_reference_differences_retain_json_evidence(tmp_path: "Path") -> None:
    import json

    from r0.modules.replay import compare_oracle_report, write_oracle_report

    expected = tmp_path / "tiny.json"
    expected.write_text('{"prediction.latent_variance.0": 1e-320}')
    output = tmp_path / "failed.json"
    config = Path(__file__).resolve().parents[2] / "configs/oracles/gp-prior.toml"
    report = write_oracle_report(config, output, expected)
    verification = report["verification"]
    assert isinstance(verification, dict)
    assert verification["status"] == "FAIL"
    assert (
        json.loads(output.read_text())["verification"]["comparisons"][0]["relative_error"] is None
    )
    # Finite endpoints may also overflow subtraction and allowed-error arithmetic.
    extreme = compare_oracle_report({"value": -1e308}, {"value": 1e308}, atol=0.0, rtol=1e308)
    assert extreme["status"] == "PASS"
    assert "Infinity" not in json.dumps(extreme, allow_nan=False)
