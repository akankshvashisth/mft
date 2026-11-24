from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.adapters.extract import instrument_info, option_spec_from_instrument, plain_settings, plain_value
from src.services.portfolio import OPTION, OPTION_SPREAD, OTHER
from src.services.regime import RegimeSettings


class FakeId:
    """Anything with a string form, like a Nautilus InstrumentId."""

    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


def test_plain_value_conversions():
    assert plain_value(FakeId("SPY.ARCA")) == "SPY.ARCA"
    assert plain_value([FakeId("A.X"), FakeId("B.X")]) == ("A.X", "B.X")
    assert plain_value((690.0, True, None, "x", 3)) == (690.0, True, None, "x", 3)


def test_plain_settings_copies_matching_attributes():
    config = SimpleNamespace(
        vix_id=FakeId("^VIX.CBOE"), vix3m_id=FakeId("^VIX3M.CBOE"), interval_secs=30, unrelated="ignored"
    )
    settings = plain_settings(config, RegimeSettings, use_quotes=False)
    assert settings == RegimeSettings(vix_id="^VIX.CBOE", vix3m_id="^VIX3M.CBOE", interval_secs=30, use_quotes=False)


def test_plain_settings_requires_mandatory_fields():
    @dataclass(frozen=True)
    class Needs:
        mandatory: str

    with pytest.raises(TypeError):
        plain_settings(SimpleNamespace(), Needs)


def option(**overrides):
    fields = dict(
        id=FakeId("SPY 261218C00720000.SMART"), underlying="SPY", strike_price=720.0, option_kind="OptionKind.CALL",
        expiration_ns=1_797_000_000_000_000_000, multiplier=100, info={"open_interest": "1500"},
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_option_spec_from_instrument():
    spec = option_spec_from_instrument(option())
    assert (spec.instrument_id, spec.underlying, spec.strike, spec.is_call) == ("SPY 261218C00720000.SMART", "SPY", 720.0, True)
    assert (spec.expiry_ns, spec.multiplier, spec.open_interest) == (1_797_000_000_000_000_000, 100.0, 1500.0)
    assert not option_spec_from_instrument(option(option_kind="PUT")).is_call


@pytest.mark.parametrize("info, oi", [(None, None), ({}, None), ({"open_interest": "n/a"}, None), ("junk", None)])
def test_unknown_open_interest_stays_unknown(info, oi):
    assert option_spec_from_instrument(option(info=info)).open_interest is oi


def test_instrument_info_defaults():
    opt = instrument_info(option(), OPTION)
    assert (opt.kind, opt.multiplier, opt.option.strike) == (OPTION, 100.0, 720.0)
    assert instrument_info(SimpleNamespace(multiplier=None), OPTION_SPREAD).multiplier == 100.0
    assert instrument_info(SimpleNamespace(), OTHER).multiplier == 1.0
    future = instrument_info(SimpleNamespace(multiplier=5), OTHER)
    assert (future.kind, future.multiplier, future.option) == (OTHER, 5.0, None)
