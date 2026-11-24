"""Quote validity and combo pricing helpers."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


def two_sided_mid(bid: float, ask: float) -> float | None:
    """Mid of a usable two-sided quote (both sides positive, not crossed); None otherwise."""
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        return None
    return 0.5 * (bid + ask)


def is_tradable_quote(bid: float, ask: float) -> bool:
    """Positive bid with a strictly positive spread; locked or crossed books are not tradable."""
    return bid > 0.0 and ask > bid


def combo_mid_natural(
    legs: Iterable[Mapping[str, Any]],
    quote_lookup: Callable[[str], tuple[float, float] | None],
    selling: bool,
) -> tuple[float, float] | None:
    """(mid, natural) of a combo defined as sum(ratio x leg price).

    Legs are mappings with "instrument_id" and "ratio". Selling the combo hits bids on positive-ratio
    legs and lifts offers on negative-ratio legs; buying mirrors that. None if any leg has no quote.
    """
    mid = 0.0
    natural = 0.0
    for leg in legs:
        quote = quote_lookup(str(leg["instrument_id"]))
        if quote is None:
            return None
        bid, ask = quote
        ratio = int(leg["ratio"])
        mid += ratio * 0.5 * (bid + ask)
        if selling:
            natural += ratio * (bid if ratio > 0 else ask)
        else:
            natural += ratio * (ask if ratio > 0 else bid)
    return mid, natural
