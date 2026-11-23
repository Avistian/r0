from r0.modules.episodes import Episode, Feature, Row
from r0.modules.featurize import transform


def test_context_statistics_and_unknown_missing_are_distinct() -> None:
    episode = Episode(
        "task",
        "ep",
        "regression",
        (Feature("x", "continuous"), Feature("color", "categorical")),
        (Row("a", (1.0, "red")), Row("b", (3.0, "blue"))),
        (0.0, 2.0),
        (Row("c", (9.0, "green")), Row("d", (None, None))),
        (3.0, 4.0),
    )
    result = transform(episode.predictor_view())
    assert result.context[0].values == (-1.0, 1)
    assert result.queries[0].values == (7.0, -1)
    assert result.queries[0].unknown == (False, True)
    assert result.queries[1].missing == (True, True)
    assert result.queries[1].unknown == (False, False)
    assert result.statistics[0].mean == 2.0
    assert result.statistics[0].scale == 1.0


def test_query_values_do_not_fit_statistics_and_raw_rows_remain_immutable() -> None:
    from dataclasses import FrozenInstanceError, replace

    import pytest

    episode = Episode(
        "task",
        "ep",
        "regression",
        (
            Feature("x", "continuous"),
            Feature("constant", "count"),
            Feature("empty", "continuous"),
            Feature("time", "timestamp", unit="s", timezone="UTC"),
        ),
        (Row("c", (2.0, 3, None, 100)),),
        (1.0,),
        (Row("q", (8.0, 4, 9.0, 200)),),
        (2.0,),
    )
    result = transform(episode.predictor_view())
    changed = replace(episode, queries=(Row("q", (800.0, 400, 90.0, 999)),))
    assert result.statistics == transform(changed.predictor_view()).statistics
    assert result.statistics[1].constant
    assert result.statistics[2].all_missing
    assert not result.statistics[2].constant
    assert result.queries[0].values == (6.0, 1.0, 9.0, 200)
    field = "values"
    with pytest.raises(FrozenInstanceError):
        setattr(episode.predictor_view().context[0], field, (99.0,))
    assert episode.context[0].values == (2.0, 3, None, 100)


def test_representable_large_context_statistics() -> None:
    episode = Episode(
        "task",
        "ep",
        "regression",
        (Feature("x", "continuous"),),
        tuple(Row(str(i), (1e308,)) for i in range(3)),
        (1.0, 2.0, 3.0),
        (Row("q", (1e308,)),),
        (4.0,),
    )
    result = transform(episode.predictor_view())
    assert result.statistics[0].mean == 1e308
    assert result.queries[0].values == (0.0,)
