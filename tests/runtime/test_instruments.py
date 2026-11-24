from src.runtime.instruments import (
    RV_UNIVERSE,
    SPY_DEC26_SLICE,
    SYNC_UNIVERSE,
    ib_future_contract,
    ib_option_contracts,
    occ_option_id,
    slice_option_ids,
    split_instrument_id,
)
from src.services.vol_surface import VolSurfaceSettings


def test_occ_option_ids():
    assert occ_option_id("SPY", "261218", False, 690.0) == "SPY 261218P00690000.SMART"
    assert occ_option_id("SPY", "261218", True, 722.5) == "SPY 261218C00722500.SMART"
    assert occ_option_id("QQQ", "261218", True, 500.0, venue="CBOE") == "QQQ 261218C00500000.CBOE"


def test_spy_slice_matches_what_is_loaded():
    s = SPY_DEC26_SLICE
    assert s.put_strikes == (685.0, 690.0, 695.0, 700.0, 705.0, 710.0, 715.0, 720.0)
    assert s.call_strikes == (720.0, 725.0, 730.0, 735.0, 740.0, 745.0, 750.0, 755.0)
    assert (s.expiry_yymmdd, s.strike_min, s.strike_max) == ("261218", 685.0, 755.0)


def test_option_ids_and_ib_contracts_come_from_the_same_slice():
    ids = slice_option_ids(SPY_DEC26_SLICE)
    contracts = ib_option_contracts(SPY_DEC26_SLICE)

    assert (len(ids), ids[0], ids[-1]) == (16, "SPY 261218P00685000.SMART", "SPY 261218C00755000.SMART")
    assert [
        occ_option_id(c["symbol"], c["lastTradeDateOrContractMonth"][2:], c["right"] == "C", c["strike"]) for c in contracts
    ] == list(ids)
    assert contracts[0] == {
        "secType": "OPT", "symbol": "SPY", "exchange": "SMART",
        "lastTradeDateOrContractMonth": "20261218", "strike": 685.0, "right": "P",
    }


def test_vol_actor_default_strike_range_covers_the_slice():
    assert (VolSurfaceSettings.strike_min, VolSurfaceSettings.strike_max) == (
        SPY_DEC26_SLICE.strike_min,
        SPY_DEC26_SLICE.strike_max,
    )


def test_future_contract_and_universes():
    assert ib_future_contract("MES") == {
        "secType": "FUT", "symbol": "MES", "exchange": "CME", "currency": "USD", "lastTradeDateOrContractMonth": "202612",
    }
    assert len(RV_UNIVERSE) == 8 and RV_UNIVERSE[0] == "SPY.ARCA"
    assert SYNC_UNIVERSE == RV_UNIVERSE + ("VIXY.ARCA", "VIXM.ARCA")
    assert split_instrument_id("SMH.NASDAQ") == ("SMH", "NASDAQ")
