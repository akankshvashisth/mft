"""Numeric, time and infrastructure constants shared across the system."""

NS_PER_SEC: int = 1_000_000_000
SECS_PER_DAY: float = 86_400.0
NS_PER_DAY: int = 86_400 * NS_PER_SEC
DAYS_PER_YEAR: float = 365.25
NS_PER_YEAR: float = DAYS_PER_YEAR * NS_PER_DAY
TRADING_DAYS_PER_YEAR: int = 252

# Payload timestamps at or below this value are not epoch nanoseconds (zero, seconds, ...)
MIN_EPOCH_NS: float = 1e15

DEFAULT_RISK_FREE_RATE: float = 0.0425
DEFAULT_VALKEY_URL: str = "redis://127.0.0.1:6379/0"
NY_TZ_NAME: str = "America/New_York"
