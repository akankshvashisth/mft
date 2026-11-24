from nautilus_trader.model import (
    CustomData,
    DataType,
    InstrumentId,
    register_custom_data_class,
)
from nautilus_trader.model.custom import customdataclass


@customdataclass
class SviSliceParams:
    """Calibrated volatility surface slice & static arbitrage state."""
    underlying_id: InstrumentId
    expiry_ns: int
    forward: float
    atm_iv: float
    skew_25d: float
    arb_free: bool
    quote_count: int


@customdataclass
class MacroRegimeSignal:
    """VIX / VIX3M term-structure regime gate."""
    vix: float
    vix3m: float
    ratio: float
    short_vol_allowed: bool


register_custom_data_class(SviSliceParams)
register_custom_data_class(MacroRegimeSignal)

@customdataclass
class RealizedVolSnapshot:
    """Live P-measure realized volatility, jump ratio, and cross-asset correlation."""
    underlying_id: InstrumentId
    spot: float
    rv_5d: float
    rv_10d: float
    rv_20d: float
    rv_60d: float
    rv_percentile: float
    jump_ratio: float
    avg_pair_corr: float


register_custom_data_class(RealizedVolSnapshot)
