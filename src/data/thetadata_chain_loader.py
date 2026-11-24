from __future__ import annotations
import math

from datetime import date
from io import StringIO
from pathlib import Path
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd


class ThetaDataChainLoader:
    """Fetches option chain snapshots / historical EOD & 1-min quotes from local ThetaTerminal v3."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:25503",
        out_dir: str | Path = "data/thetadata/chains",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def fetch_option_eod_chain(
        self,
        root: str,
        start_date: str,
        end_date: str,
        expiration: str = "*",
    ) -> pd.DataFrame:
        """Pulls historical option EOD/quote chain via ThetaTerminal v3 and caches to Parquet."""
        params = urllib.parse.urlencode(
            {
                "symbol": root.upper(),
                "expiration": expiration,
                "start_date": start_date.replace("-", ""),
                "end_date": end_date.replace("-", ""),
                "format": "csv",
            }
        )
        url = f"{self.base_url}/v3/option/history/eod?{params}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            csv_text = resp.read().decode("utf-8")

        if not csv_text.strip():
            return pd.DataFrame()

        df = pd.read_csv(StringIO(csv_text))
        df.columns = [c.strip().lower() for c in df.columns]
        if "strike" in df.columns and df["strike"].max() > 10_000:
            # ThetaData sometimes encodes strikes in thousandths of a dollar
            df["strike"] = df["strike"] / 1000.0

        out_path = self.out_dir / f"{root.upper()}_{start_date}_{end_date}.parquet"
        df.to_parquet(out_path, index=False)
        return df

    @staticmethod
    def build_synthetic_midcap_chain(
        symbol: str = "MIDX",
        start_price: float = 85.0,
        n_days: int = 120,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generates a realistic American mid-cap option chain replay dataset for offline testing."""
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(end=date.today(), periods=n_days)
        spots = [start_price]
        realized_vols = [0.28]

        for _ in range(1, n_days):
            rv = float(np.clip(realized_vols[-1] + rng.normal(0.0, 0.012), 0.16, 0.55))
            ret = rng.normal(0.06 / 252.0, rv / np.sqrt(252.0))
            spots.append(spots[-1] * np.exp(ret))
            realized_vols.append(rv)

        rows = []
        for dt, spot, rv in zip(dates, spots, realized_vols):
            # Implied vol carries a realistic +4.5 vol pt VRP over realized vol
            atm_iv = rv + 0.045 + rng.normal(0.0, 0.008)
            strikes = np.arange(round(spot * 0.82), round(spot * 1.18) + 1.0, 2.5)
            ttm = 30.0 / 365.25
            for k in strikes:
                m = np.log(k / spot)
                iv = max(0.10, atm_iv - 0.35 * m + 0.90 * (m ** 2))
                for right in ("P", "C"):
                    d1 = (np.log(spot / k) + 0.5 * iv * iv * ttm) / (iv * np.sqrt(ttm))
                    d2 = d1 - iv * np.sqrt(ttm)
                    nd1 = 0.5 * (1.0 + math.erf(d1 / np.sqrt(2.0)))
                    nd2 = 0.5 * (1.0 + math.erf(d2 / np.sqrt(2.0)))
                    if right == "C":
                        mid = max(0.05, spot * nd1 - k * np.exp(-0.0425 * ttm) * nd2)
                    else:
                        mid = max(0.05, k * np.exp(-0.0425 * ttm) * (1.0 - nd2) - spot * (1.0 - nd1))
                    # Realistic mid-cap bid-ask spread (~6-10% of mid, min $0.05)
                    half_sp = max(0.025, mid * 0.04)
                    rows.append(
                        {
                            "date": dt.date().isoformat(),
                            "symbol": symbol,
                            "spot": round(spot, 2),
                            "rv_20d": round(rv, 4),
                            "atm_iv": round(atm_iv, 4),
                            "strike": float(k),
                            "right": right,
                            "dte": 30,
                            "bid": round(max(0.01, mid - half_sp), 2),
                            "ask": round(mid + half_sp, 2),
                            "mid": round(mid, 2),
                            "iv": round(iv, 4),
                            "open_interest": int(rng.integers(250, 2500)),
                        }
                    )
        return pd.DataFrame(rows)
