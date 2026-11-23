from dataclasses import replace

import pytest

from r0.modules.episodes import Episode, Feature, Row
from r0.modules.oracles import ClassificationSpec, classification, classification_prior


def coin() -> Episode:
    return Episode(
        "coin",
        "coin-episode",
        "binary",
        (Feature("unused", "continuous"),),
        tuple(Row(f"c{i}", (0.0,)) for i in range(4)),
        ("H", "H", "H", "T"),
        (Row("q0", (0.0,)), Row("q1", (0.0,))),
        ("H", "T"),
        ("H", "T"),
        "declared",
    )


def test_coin_posterior_and_prior_ignore_query_labels() -> None:
    episode = coin()
    spec = ClassificationSpec("beta_bernoulli", ("H", "T"), (1.0, 1.0), "H")
    result = classification(episode.predictor_view(), spec)
    assert result.posterior == (4.0, 2.0)
    assert result.probabilities[0] == pytest.approx((2 / 3, 1 / 3))
    assert result == classification(
        replace(episode, query_targets=("T", "T")).predictor_view(), spec
    )
    assert classification_prior(spec, 2).probabilities == ((0.5, 0.5), (0.5, 0.5))


def test_multiclass_retains_absent_class_and_asymmetric_prior() -> None:
    episode = Episode(
        "multi",
        "multi",
        "multiclass",
        (Feature("unused", "continuous"),),
        (Row("c0", (0.0,)), Row("c1", (0.0,))),
        ("A", "A"),
        (Row("q", (0.0,)),),
        ("C",),
        ("A", "B", "C"),
        "declared",
    )
    spec = ClassificationSpec("dirichlet_categorical", ("A", "B", "C"), (1.0, 2.0, 3.0))
    prediction = classification(episode.predictor_view(), spec)
    assert prediction.posterior == (3.0, 2.0, 3.0)
    assert prediction.probabilities == ((0.375, 0.25, 0.375),)
    with pytest.raises(ValueError, match="class order"):
        classification(replace(episode, classes=("B", "A", "C")).predictor_view(), spec)
    with pytest.raises(ValueError, match="strictly positive"):
        replace(spec, concentration=(0.0, 1.0, 2.0))
