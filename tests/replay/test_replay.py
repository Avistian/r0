from dataclasses import replace

import pytest

from r0.modules.replay import Config, build_episode, parse_config, replay


def test_replay_identity_and_configuration_validation() -> None:
    config = parse_config('version = 1\nfixture = "multiclass"\n')
    first = replay(config)
    replay(replace(config, episode_key="another"))
    assert replay(config) == first
    changed_split = build_episode(replace(config, split_seed=1))
    original = build_episode(config)
    assert changed_split.source_task_id == original.source_task_id
    assert changed_split.context != original.context
    with pytest.raises(ValueError, match="unknown"):
        parse_config('version = 1\nfixture = "regression"\ntyop = 2')
    with pytest.raises(ValueError, match="version"):
        parse_config('version = 2\nfixture = "regression"')
    assert config == Config(fixture="multiclass")


def test_hand_computed_scores_and_query_label_isolation() -> None:
    from r0.modules.featurize import transform
    from r0.modules.replay import predict, score

    episode = build_episode(Config(fixture="regression"))
    predictions = predict(transform(episode.predictor_view()))
    assert predictions == ((5.0,), (5.0,), (5.0,))
    assert score(episode, predictions)["value"] == pytest.approx(68 / 3)
    changed = replace(episode, query_targets=(0.0, 0.0, 0.0))
    assert predict(transform(changed.predictor_view())) == predictions
    assert score(changed, predictions)["value"] == 25.0
    multi = build_episode(Config(fixture="multiclass", split_seed=3))
    probabilities = predict(transform(multi.predictor_view()))
    assert probabilities == ((1 / 3, 2 / 3, 0.0),) * 3
    assert score(multi, probabilities)["value"] == pytest.approx(10 / 9)


@pytest.mark.parametrize(
    "field",
    [
        "root_seed = true",
        "context_rows = 0",
        "split_seed = -1",
        'preprocessing = "unknown"',
        "root_seed = 1.0",
    ],
)
def test_malformed_config_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="expected|must be|unsupported"):
        parse_config('version = 1\nfixture = "binary"\n' + field)


def test_formatting_and_explicit_defaults_preserve_fingerprints() -> None:
    first = replay(parse_config('version=1\nfixture="regression"'))
    second = replay(parse_config('fixture = "regression"\nversion = 1\nroot_seed = 0'))
    assert first["config_fingerprint"] == second["config_fingerprint"]
    assert first["content_fingerprint"] == second["content_fingerprint"]


def test_scoring_rejects_invalid_prediction_values() -> None:
    from r0.modules.replay import score

    episode = build_episode(Config(fixture="regression"))
    with pytest.raises(ValueError, match="finite"):
        score(episode, ((float("nan"),),) * 3)
    with pytest.raises(ValueError, match="width"):
        score(episode, ((1.0, 2.0),) * 3)


def test_out_of_range_score_has_actionable_error() -> None:
    from r0.modules.replay import score

    episode = build_episode(Config(fixture="regression"))
    with pytest.raises(ValueError, match="numeric range"):
        score(episode, ((1e200,),) * 3)
