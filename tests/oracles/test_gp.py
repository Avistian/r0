import pytest

from r0.modules.episodes import Episode, Feature, Row
from r0.modules.oracles import GPSpec, gp, gp_prior


def test_one_observation_linear_gp_has_independent_hand_answer() -> None:
    # k(1,1)=1, noise=1, y=2: mean(2)=2, latent var(2)=2, observed var=3.
    episode = Episode(
        "linear",
        "one",
        "regression",
        (Feature("x", "continuous"),),
        (Row("c", (1.0,)),),
        (2.0,),
        (Row("q", (2.0,)),),
        (3.0,),
    )
    spec = GPSpec("linear", noise_variance=1.0)
    result = gp(episode.predictor_view(), spec)
    assert result.mean == pytest.approx((2.0,))
    assert result.latent_variance == pytest.approx((2.0,))
    assert result.observation_variance == pytest.approx((3.0,))
    assert result.conditioning_covariance == ((2.0,),)
    prior = gp_prior((2.0,), spec)
    assert prior.mean == (0.0,)
    assert prior.latent_covariance == ((4.0,),)


def test_rbf_near_far_queries_and_explicit_jitter() -> None:
    from dataclasses import replace

    episode = Episode(
        "rbf",
        "one",
        "regression",
        (Feature("x", "continuous"),),
        (Row("c", (0.0,)),),
        (2.0,),
        (Row("near", (0.0,)), Row("far", (8.0,))),
        (1.0, 0.0),
    )
    spec = GPSpec("rbf", length_scale=1.0, noise_variance=1.0)
    result = gp(episode.predictor_view(), spec)
    assert result.mean == pytest.approx((1.0, 0.0), abs=1e-10)
    assert result.latent_variance == pytest.approx((0.5, 1.0))
    assert result.observation_variance == pytest.approx((1.5, 2.0))
    stabilized = gp(episode.predictor_view(), replace(spec, noise_variance=0.0, jitter=1.0))
    assert stabilized.mean == result.mean
    assert stabilized.observation_variance == result.latent_variance
    assert stabilized.stabilized
    reordered = gp(replace(episode, queries=episode.queries[::-1]).predictor_view(), spec)
    assert reordered.mean == pytest.approx(result.mean[::-1])
    assert reordered.latent_variance == pytest.approx(result.latent_variance[::-1])


def test_duplicate_coordinates_require_solvable_conditioning() -> None:
    episode = Episode(
        "rbf",
        "duplicate",
        "regression",
        (Feature("x", "continuous"),),
        (Row("c0", (0.0,)), Row("c1", (0.0,))),
        (1.0, 3.0),
        (Row("q", (0.0,)),),
        (2.0,),
    )
    with pytest.raises(ValueError, match="conditioning failed"):
        gp(episode.predictor_view(), GPSpec("rbf", length_scale=1.0))
    result = gp(episode.predictor_view(), GPSpec("rbf", length_scale=1.0, noise_variance=1.0))
    assert result.mean == pytest.approx((4 / 3,))
    assert result.latent_variance == pytest.approx((1 / 3,))


def test_gp_rejects_unsupported_schema_and_nonfinite_parameters() -> None:
    from dataclasses import replace

    episode = Episode(
        "bad",
        "bad",
        "regression",
        (Feature("x", "continuous"),),
        (Row("c", (None,)),),
        (1.0,),
        (Row("q", (0.0,)),),
        (0.0,),
    )
    with pytest.raises(ValueError, match="fully observed"):
        gp(episode.predictor_view(), GPSpec("linear"))
    with pytest.raises(ValueError, match="continuous"):
        gp(replace(episode, features=(Feature("x", "count"),)).predictor_view(), GPSpec("linear"))
    with pytest.raises(ValueError, match="finite"):
        GPSpec("linear", noise_variance=float("nan"))
    with pytest.raises(ValueError, match="64"):
        gp_prior((0.0,) * 65, GPSpec("linear"))


def test_prior_api_rejects_jitter_without_conditioning_system() -> None:
    with pytest.raises(ValueError, match="prior.*jitter"):
        gp_prior((0.0,), GPSpec("rbf", length_scale=1.0, jitter=1.0))
