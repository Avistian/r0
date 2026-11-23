from dataclasses import replace

import pytest

from r0.modules.episodes import Episode, Feature, Row


def test_episode_isolates_scoring_and_rejects_overlap() -> None:
    episode = Episode(
        source_task_id="task",
        episode_id="episode",
        task_type="regression",
        features=(Feature("x", "continuous"),),
        context=(Row("c", (1.0,)),),
        context_targets=(2.0,),
        queries=(Row("q", (3.0,)),),
        query_targets=(4.0,),
    )
    assert episode.predictor_view() == replace(episode, query_targets=(9.0,)).predictor_view()
    assert not hasattr(episode.predictor_view(), "query_targets")
    with pytest.raises(ValueError, match="row identities"):
        replace(episode, queries=(Row("c", (3.0,)),))


@pytest.mark.parametrize(
    ("kind", "value"), [("count", -1), ("binary", 2), ("continuous", float("inf"))]
)
def test_invalid_feature_values_have_errors(kind: str, value: float) -> None:
    from typing import cast

    from r0.modules.episodes import FeatureKind

    with pytest.raises(ValueError, match="must be"):
        Episode(
            "task",
            "ep",
            "regression",
            (Feature("x", cast(FeatureKind, kind)),),
            (Row("c", (value,)),),
            (1.0,),
            (Row("q", (1,)),),
            (2.0,),
        )


def test_declared_classes_allow_absent_context_but_reject_unknown_targets() -> None:
    episode = Episode(
        "task",
        "ep",
        "multiclass",
        (Feature("x", "binary"),),
        (Row("c", (0,)),),
        ("A",),
        (Row("q", (1,)),),
        ("C",),
        ("A", "B", "C"),
        "declared fixture",
    )
    assert episode.predictor_view().classes == ("A", "B", "C")
    with pytest.raises(ValueError, match="outside declared"):
        replace(episode, query_targets=("D",))
    with pytest.raises(ValueError, match="finite numbers"):
        replace(
            episode,
            task_type="regression",
            classes=(),
            class_provenance=None,
            context_targets=(1.0,),
            query_targets=(float("nan"),),
        )


def test_predictor_owns_records_and_provenance_is_a_string() -> None:
    from typing import cast

    from r0.modules.replay import Config, build_episode

    episode = build_episode(Config(fixture="binary"))
    view = episode.predictor_view()
    assert view.context[0] is not episode.context[0]
    assert view.features[0] is not episode.features[0]
    with pytest.raises(ValueError, match="provenance"):
        replace(episode, class_provenance=cast(str, ["mutable"]))
