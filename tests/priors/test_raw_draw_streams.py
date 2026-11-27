"""Nested function fits commit captured construction and fit streams together."""

import copy

import numpy as np
import pytest

from r0.modules import priors


def test_multiple_streams_share_one_ordered_transcript_and_nested_single_stream() -> None:
    originals = {"raw": np.random.default_rng(3), "fit": np.random.default_rng(4)}
    controls = copy.deepcopy(originals)
    capture = priors.RawDrawLedger()
    with capture.transactions(originals) as (streams, ledger):
        ledger.draw(
            streams["fit"],
            owner=("node",),
            role="rows",
            distribution="integers",
            shape=(3,),
            parameters=(("high", 11),),
        )
        with ledger.transaction(streams["raw"]) as (raw, nested):
            nested.draw(
                raw, owner=("node", "matrix"), role="weights", distribution="normal", shape=(2, 3)
            )
        ledger.draw(
            streams["fit"],
            owner=("node",),
            role="order",
            distribution="permutation",
            shape=(5,),
            parameters=(("count", 5),),
        )
    records = capture.freeze()
    assert [record.stream for record in records] == ["fit", "raw", "fit"]
    for record in records:
        np.testing.assert_array_equal(record.replay(controls[record.stream]), record.array())
    for name in originals:
        np.testing.assert_array_equal(originals[name].random(8), controls[name].random(8))


def test_multiple_stream_failure_rolls_back_all_generators_and_records() -> None:
    originals = {"raw": np.random.default_rng(3), "fit": np.random.default_rng(4)}
    controls = copy.deepcopy(originals)
    capture = priors.RawDrawLedger(limits=priors.RawDrawLimits(numbers=8))

    def attempt() -> None:
        with capture.transactions(originals) as (streams, ledger):
            ledger.draw(
                streams["raw"], owner=("node",), role="weights", distribution="normal", shape=(8,)
            )
            ledger.draw(
                streams["fit"],
                owner=("node",),
                role="rows",
                distribution="integers",
                shape=(1,),
                parameters=(("high", 3),),
            )

    with pytest.raises(ValueError, match="raw draw.*limit"):
        attempt()
    assert capture.freeze() == ()
    for name in originals:
        np.testing.assert_array_equal(originals[name].random(8), controls[name].random(8))


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("mapping", ["caller", "staged"])
@pytest.mark.parametrize("mutation", ["replace", "delete", "add"])
def test_transaction_rejects_stream_mapping_mutation(
    captured: bool, mapping: str, mutation: str
) -> None:
    from r0.modules.priors.lib.raw_draw_capture import raw_transactions

    originals = {"raw": np.random.default_rng(3), "fit": np.random.default_rng(4)}
    retained = originals.copy()
    controls = copy.deepcopy(originals)
    capture = priors.RawDrawLedger() if captured else None

    def attempt() -> None:
        with raw_transactions(originals, capture) as (streams, ledger):
            if ledger is None:
                streams["raw"].normal(size=4)
            else:
                ledger.draw(
                    streams["raw"],
                    owner=("node",),
                    role="weights",
                    distribution="normal",
                    shape=(4,),
                )
            target = originals if mapping == "caller" else streams
            if mutation == "delete":
                del target["fit"]
            elif mutation == "add":
                target["extra"] = np.random.default_rng(5)
            else:
                target["fit"] = copy.deepcopy(target["fit"])

    with pytest.raises(ValueError, match="stream bindings changed"):
        attempt()
    if capture is not None:
        assert capture.freeze() == ()
    for name in retained:
        np.testing.assert_array_equal(retained[name].random(8), controls[name].random(8))


def test_multiple_streams_reject_aliases_before_any_draw() -> None:
    rng = np.random.default_rng(3)
    control = copy.deepcopy(rng)
    capture = priors.RawDrawLedger()

    def attempt() -> None:
        with capture.transactions({"raw": rng, "fit": np.random.Generator(rng.bit_generator)}):
            pytest.fail("aliased generators were admitted")

    with pytest.raises(ValueError, match="distinct bit generators"):
        attempt()
    assert capture.freeze() == ()
    np.testing.assert_array_equal(rng.random(8), control.random(8))


def test_multi_stream_external_mutation_is_preserved_and_other_stream_is_unchanged() -> None:
    originals = {"raw": np.random.default_rng(3), "fit": np.random.default_rng(4)}
    controls = copy.deepcopy(originals)
    capture = priors.RawDrawLedger()

    def attempt() -> None:
        with capture.transactions(originals) as (streams, ledger):
            ledger.draw(
                streams["raw"], owner=("node",), role="weights", distribution="normal", shape=(4,)
            )
            originals["fit"].random(3)

    with pytest.raises(ValueError, match="caller RNG changed"):
        attempt()
    controls["fit"].random(3)
    assert capture.freeze() == ()
    for name in originals:
        np.testing.assert_array_equal(originals[name].random(8), controls[name].random(8))


def test_multi_stream_rejects_false_stream_label_without_drawing() -> None:
    originals = {"raw": np.random.default_rng(3), "fit": np.random.default_rng(4)}
    controls = copy.deepcopy(originals)
    capture = priors.RawDrawLedger()

    def attempt() -> None:
        with capture.transactions(originals) as (streams, ledger):
            ledger.draw(
                streams["raw"],
                owner=("node",),
                role="weights",
                distribution="normal",
                shape=(4,),
                stream="fit",
            )

    with pytest.raises(ValueError, match="label differs"):
        attempt()
    assert capture.freeze() == ()
    for name in originals:
        np.testing.assert_array_equal(originals[name].random(8), controls[name].random(8))
