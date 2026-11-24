import pytest

from src.dashboard.formatting import fmt, pct_or_none


@pytest.mark.parametrize(
    "val, spec, text",
    [(1.2345, ".2f", "1.23"), (0.0123, ".2%", "1.23%"), (-3, "+d", "-3"), (None, ".2f", "N/A"), (True, ".2f", "N/A"), ("7", ".2f", "N/A")],
)
def test_fmt(val, spec, text):
    assert fmt(val, spec) == text


def test_fmt_custom_placeholder():
    assert fmt(None, ".2f", na="-") == "-"


def test_pct_or_none():
    assert pct_or_none(0.0425) == 4.25
    assert pct_or_none(0.123456, 1) == 12.3
    assert pct_or_none(None) is None
    assert pct_or_none(False) is None
