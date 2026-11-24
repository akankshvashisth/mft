import pytest

from src.common.quotes import combo_mid_natural, is_tradable_quote, two_sided_mid


@pytest.mark.parametrize(
    "bid, ask, expected",
    [
        (1.0, 1.2, 1.1),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, None),
        (1.0, 0.0, None),
        (-1.0, 1.0, None),
        (1.2, 1.0, None),
    ],
)
def test_two_sided_mid(bid, ask, expected):
    mid = two_sided_mid(bid, ask)
    if expected is None:
        assert mid is None
    else:
        assert mid == pytest.approx(expected)


@pytest.mark.parametrize(
    "bid, ask, tradable",
    [(1.0, 1.2, True), (1.0, 1.0, False), (0.0, 1.0, False), (1.2, 1.0, False)],
)
def test_tradable_quote_needs_a_positive_spread(bid, ask, tradable):
    assert is_tradable_quote(bid, ask) is tradable


LEGS = [{"instrument_id": "SOLD", "ratio": 1}, {"instrument_id": "WING", "ratio": -1}]
QUOTES = {"SOLD": (2.0, 2.2), "WING": (0.9, 1.0)}


def test_selling_a_credit_spread_hits_the_sold_leg_bid_and_lifts_the_wing_offer():
    mid, natural = combo_mid_natural(LEGS, QUOTES.get, selling=True)

    assert mid == pytest.approx(2.1 - 0.95)
    assert natural == pytest.approx(2.0 - 1.0)
    assert natural <= mid


def test_buying_back_pays_the_sold_leg_offer_and_sells_the_wing_bid():
    mid, natural = combo_mid_natural(LEGS, QUOTES.get, selling=False)

    assert mid == pytest.approx(1.15)
    assert natural == pytest.approx(2.2 - 0.9)
    assert natural >= mid


def test_missing_leg_quote_gives_no_price():
    assert combo_mid_natural(LEGS, {"SOLD": (2.0, 2.2)}.get, selling=True) is None
