"""
fetch.py — Data fetchers for Kalshi, Polymarket, Binance, and Deribit.

All functions return DataFrames and save to data/raw/ as parquet.
Timestamps are always UTC.
"""

import os
import re
import time
import warnings
import requests
# Suppress LibreSSL/OpenSSL version mismatch warning from urllib3
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ─────────────────────────────────────────────
# KALSHI
# ─────────────────────────────────────────────

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def kalshi_get(endpoint: str, params: dict = None, api_key: str = None) -> dict:
    """Simple GET wrapper with optional auth header."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(f"{KALSHI_BASE}{endpoint}", params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()


def fetch_all_kalshi_markets(api_key: str = None, status: str = None) -> pd.DataFrame:
    """
    Pull the full Kalshi market catalog via GET /markets (paginated).
    Returns a DataFrame with one row per market.

    status: 'open', 'closed', 'settled', or None for all
    """
    markets = []
    cursor = None
    page = 0

    while True:
        params = {"limit": 1000}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        data = kalshi_get("/markets", params=params, api_key=api_key)
        batch = data.get("markets", [])
        if not batch:
            break

        markets.extend(batch)
        cursor = data.get("cursor")
        page += 1
        print(f"  page {page}: fetched {len(batch)} markets (total {len(markets)})")

        if not cursor:
            break

    df = pd.DataFrame(markets)
    return df


def fetch_kalshi_historical_markets(api_key: str = None) -> pd.DataFrame:
    """Pull settled/historical markets from GET /historical/markets (paginated)."""
    markets = []
    cursor = None
    page = 0

    while True:
        params = {"limit": 1000}
        if cursor:
            params["cursor"] = cursor

        data = kalshi_get("/historical/markets", params=params, api_key=api_key)
        batch = data.get("markets", [])
        if not batch:
            break

        markets.extend(batch)
        cursor = data.get("cursor")
        page += 1
        print(f"  historical page {page}: fetched {len(batch)} (total {len(markets)})")

        if not cursor:
            break

    return pd.DataFrame(markets)


def _series_ticker_from_market_ticker(ticker: str) -> str:
    """
    Derive the series ticker from a market ticker.
    e.g. 'KXBTCRESERVE-27-JAN01' -> 'KXBTCRESERVE'
         'KXCPI-26APR-T0.0'      -> 'KXCPI'
         'KXBTCMAX100-26JAN01'   -> 'KXBTCMAX100'
    """
    m = re.match(r'^(.*?)-\d{2}', ticker)
    return m.group(1) if m else ticker


def fetch_kalshi_candlesticks(
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1,
    is_historical: bool = False,
    api_key: str = None,
    series_ticker: str = None,
) -> pd.DataFrame:
    """
    Fetch 1-minute candlestick data for a single Kalshi market.

    ticker          : market ticker string (e.g. 'KXBTCRESERVE-27-JAN01')
    start_ts        : Unix timestamp seconds (UTC)
    end_ts          : Unix timestamp seconds (UTC)
    period_interval : 1 = 1min, 60 = 1hr, 1440 = 1day
    is_historical   : unused (kept for API compatibility); Kalshi uses one endpoint
    series_ticker   : optional override; derived automatically from ticker if not provided

    Returns DataFrame with columns:
        timestamp_utc, yes_bid, yes_ask, volume, open_interest, prob_mid, spread_pct
    """
    if series_ticker is None:
        series_ticker = _series_ticker_from_market_ticker(ticker)
    endpoint = f"/series/{series_ticker}/markets/{ticker}/candlesticks"

    # Kalshi enforces a max time window per request (~80h for 1-min, larger for coarser intervals).
    # Chunk the range so we never exceed 72h for minute-level data.
    if period_interval <= 60:
        chunk_secs = 72 * 3600  # 72-hour windows
    else:
        chunk_secs = end_ts - start_ts  # no chunking needed for daily bars

    all_candles = []
    chunk_start = start_ts

    while chunk_start < end_ts:
        chunk_end = min(chunk_start + chunk_secs, end_ts)
        cursor = None

        while True:
            params = {
                "start_ts": chunk_start,
                "end_ts": chunk_end,
                "period_interval": period_interval,
            }
            if cursor:
                params["cursor"] = cursor

            # Retry on 429 with exponential backoff
            for attempt in range(4):
                try:
                    data = kalshi_get(endpoint, params=params, api_key=api_key)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 3:
                        time.sleep(2 ** attempt)  # 1, 2, 4 seconds
                    else:
                        raise

            candles = data.get("candlesticks", [])
            if not candles:
                break

            all_candles.extend(candles)
            cursor = data.get("cursor")
            if not cursor:
                break

        chunk_start = chunk_end
        if period_interval <= 60:
            time.sleep(0.15)  # ~6 req/s to stay under rate limit

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)

    # Normalize timestamp
    df["timestamp_utc"] = pd.to_datetime(df["end_period_ts"], unit="s", utc=True)

    # Extract price fields — Kalshi v2 nests prices inside a 'price' object
    if "price" in df.columns:
        price_df = pd.json_normalize(df["price"])
        for col in price_df.columns:
            df[col] = price_df[col]

    # Extract yes_bid / yes_ask — each is a nested dict with close/open/high/low_dollars
    for side in ("yes_bid", "yes_ask"):
        if side in df.columns and df[side].dtype == object:
            nested = pd.json_normalize(df[side].tolist())
            # Prefer close_dollars; fall back to mean_dollars
            for price_key in ("close_dollars", "mean_dollars", "open_dollars"):
                col = price_key
                if col in nested.columns:
                    df[f"{side}_close"] = pd.to_numeric(nested[col], errors="coerce")
                    break
            df = df.drop(columns=[side])
            df = df.rename(columns={f"{side}_close": side})

    # Rename volume / OI columns
    rename = {
        "volume_fp": "volume",
        "open_interest_fp": "open_interest",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Compute derived columns
    if "yes_bid" in df.columns and "yes_ask" in df.columns:
        df["yes_bid"] = pd.to_numeric(df["yes_bid"], errors="coerce")
        df["yes_ask"] = pd.to_numeric(df["yes_ask"], errors="coerce")
        df["prob_mid"] = (df["yes_bid"] + df["yes_ask"]) / 2
        # spread as fraction of mid (avoid div/0)
        mid = df["prob_mid"].replace(0, np.nan)
        df["spread_pct"] = (df["yes_ask"] - df["yes_bid"]) / mid
    elif "close_dollars" in df.columns:
        # fallback: use close price from the price object as prob_mid
        df["prob_mid"] = pd.to_numeric(df.get("close_dollars"), errors="coerce")

    df["ticker"] = ticker
    cols = [
        "ticker", "timestamp_utc", "yes_bid", "yes_ask",
        "prob_mid", "spread_pct", "volume", "open_interest",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("timestamp_utc").reset_index(drop=True)

    return df


def save_kalshi_market(ticker: str, df: pd.DataFrame) -> Path:
    """Save a market's candlestick data to data/raw/kalshi/{ticker}.parquet"""
    out = RAW_DIR / "kalshi" / f"{ticker}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load_kalshi_market(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / "kalshi" / f"{ticker}.parquet"
    return pd.read_parquet(path)


# ─────────────────────────────────────────────
# POLYMARKET
# ─────────────────────────────────────────────

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def fetch_polymarket_markets(keyword: str = None, limit: int = 100) -> pd.DataFrame:
    """
    Discover Polymarket markets via the Gamma API.
    Returns metadata including conditionId and clobTokenIds.

    NOTE: For resolved markets, use The Graph subgraph for price history
    (CLOB API enforces 12-hour floor on resolved market granularity).
    """
    params = {"limit": limit}
    if keyword:
        params["q"] = keyword

    resp = requests.get(f"{GAMMA_BASE}/markets", params=params)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def fetch_polymarket_prices(token_id: str, fidelity: int = 1) -> pd.DataFrame:
    """
    Fetch price history from Polymarket CLOB API.
    fidelity = 1 means 1-minute bars (only works for ACTIVE markets).
    For RESOLVED markets, this returns 12-hour bars max — use The Graph instead.

    Returns DataFrame with timestamp_utc, price (probability 0-1).
    """
    params = {
        "market": token_id,
        "interval": "all",
        "fidelity": fidelity,
    }
    resp = requests.get(f"{CLOB_BASE}/prices-history", params=params)
    resp.raise_for_status()
    data = resp.json()

    history = data.get("history", [])
    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history)
    df["timestamp_utc"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.rename(columns={"p": "prob_mid"})
    df["prob_mid"] = pd.to_numeric(df["prob_mid"], errors="coerce")
    return df[["timestamp_utc", "prob_mid"]].sort_values("timestamp_utc").reset_index(drop=True)


def fetch_polymarket_thegraph(condition_id: str, subgraph_url: str) -> pd.DataFrame:
    """
    Fetch resolved Polymarket market trade history via The Graph subgraph.
    Returns block-level (~2s) trade data for resolved markets.

    subgraph_url: The Graph endpoint for the Polymarket Orderbook subgraph
    condition_id: from Gamma API market metadata

    Returns DataFrame with timestamp_utc, price (probability).
    """
    query = """
    {
      fixedProductMarketMakers(where: {conditionIds_contains: ["%s"]}) {
        id
        tradesQuantity
        outcomeTokenPrices
        lastActiveDay
      }
      fpmmTrades(
        where: {fpmm_: {conditionIds_contains: ["%s"]}}
        orderBy: creationTimestamp
        orderDirection: asc
        first: 1000
      ) {
        id
        creationTimestamp
        outcomeIndex
        outcomeTokensTraded
        collateralAmount
        feeAmount
      }
    }
    """ % (condition_id, condition_id)

    resp = requests.post(subgraph_url, json={"query": query})
    resp.raise_for_status()
    data = resp.json()

    trades = data.get("data", {}).get("fpmmTrades", [])
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp_utc"] = pd.to_datetime(
        df["creationTimestamp"].astype(int), unit="s", utc=True
    )
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def save_polymarket_market(condition_id: str, df: pd.DataFrame) -> Path:
    out = RAW_DIR / "polymarket" / f"{condition_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


# ─────────────────────────────────────────────
# BINANCE — Crypto Spot (1-minute OHLCV)
# ─────────────────────────────────────────────

BINANCE_BASE = "https://api.binance.com"


def fetch_binance_klines(
    symbol: str,
    interval: str = "1m",
    start_ms: int = None,
    end_ms: int = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch klines directly from Binance REST API (paginated).
    Use this for recent data. For bulk historical data use binance-historical-data package.

    Returns DataFrame with timestamp_utc, open, high, low, close, volume, log_ret.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms

    all_rows = []
    while True:
        resp = requests.get(f"{BINANCE_BASE}/api/v3/klines", params=params)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        all_rows.extend(batch)
        last_ts = batch[-1][0]
        params["startTime"] = last_ts + 1

        if len(batch) < limit:
            break
        if end_ms and last_ts >= end_ms:
            break

        time.sleep(0.1)  # be polite

    cols = [
        "timestamp_ms", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"].astype(int), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["symbol"] = symbol

    out_cols = ["symbol", "timestamp_utc", "open", "high", "low", "close", "volume", "log_ret"]
    return df[out_cols].reset_index(drop=True)


def load_binance_bulk(symbol: str, data_dir: Path = None) -> pd.DataFrame:
    """
    Load crypto data downloaded via binance-historical-data package.
    Expects CSV files in data/raw/crypto/{symbol}/ directory.

    Usage:
        from binance_historical_data import BinanceDataDumper
        dumper = BinanceDataDumper(path_dir_where_to_dump="data/raw/crypto/")
        dumper.dump_klines(tickers=["BTCUSDT"], intervals=["1m"],
                           date_start=datetime(2022,1,1), date_end=datetime(2026,1,1))
    """
    if data_dir is None:
        data_dir = RAW_DIR / "crypto" / symbol / "spot" / "monthly" / "klines" / "1m"

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    cols = [
        "timestamp_ms", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ]

    dfs = []
    for f in tqdm(csv_files, desc=f"Loading {symbol}"):
        fdf = pd.read_csv(f, header=None)
        fdf.columns = cols[:len(fdf.columns)]

        # Drop non-numeric rows (embedded header rows in some files).
        ts = pd.to_numeric(fdf["timestamp_ms"], errors="coerce")
        fdf = fdf[ts.notna()].copy()
        ts = ts[ts.notna()]

        # Binance changed timestamp unit in 2025: older files use ms (~1.5e12),
        # newer files use µs (~1.7e15). Normalise to ms per file.
        if ts.median() > 1e14:
            fdf["timestamp_ms"] = (ts / 1000).astype(np.int64)
        else:
            fdf["timestamp_ms"] = ts.astype(np.int64)

        dfs.append(fdf)

    df = pd.concat(dfs, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"].astype(np.int64), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])

    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["symbol"] = symbol

    out_cols = ["symbol", "timestamp_utc", "open", "high", "low", "close", "volume", "log_ret"]
    return df[out_cols]


def save_crypto(symbol: str, df: pd.DataFrame) -> Path:
    out = RAW_DIR / "crypto" / f"crypto_1m_{symbol}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load_crypto(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "crypto" / f"crypto_1m_{symbol}.parquet")


# ─────────────────────────────────────────────
# DERIBIT — Options / DVOL
# ─────────────────────────────────────────────

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


def fetch_deribit_dvol(
    currency: str = "BTC",
    start_ts: int = None,
    end_ts: int = None,
    resolution: str = "1D",
) -> pd.DataFrame:
    """
    Fetch historical daily volatility index (DVOL) from Deribit.
    Free, no auth required.

    If start_ts is provided, uses get_volatility_index_data, which supports
    explicit historical ranges. Timestamps are Unix seconds; Deribit expects ms.
    Without start_ts, falls back to get_historical_volatility for compatibility.

    Returns DataFrame with timestamp_utc, volatility (annualized IV %).
    """
    if start_ts is None:
        resp = requests.get(
            f"{DERIBIT_BASE}/get_historical_volatility",
            params={"currency": currency},
        )
        resp.raise_for_status()
        data = resp.json()["result"]
        df = pd.DataFrame(data, columns=["timestamp_ms", "volatility"])
    else:
        if end_ts is None:
            end_ts = int(pd.Timestamp.now(tz="UTC").timestamp())
        resp = requests.get(
            f"{DERIBIT_BASE}/get_volatility_index_data",
            params={
                "currency": currency,
                "start_timestamp": start_ts * 1000,
                "end_timestamp": end_ts * 1000,
                "resolution": resolution,
            },
        )
        resp.raise_for_status()
        data = resp.json()["result"].get("data", [])
        if not data:
            return pd.DataFrame(columns=["currency", "timestamp_utc", "volatility"])
        df = pd.DataFrame(data)
        if {"timestamp", "close"}.issubset(df.columns):
            df = df.rename(columns={"timestamp": "timestamp_ms", "close": "volatility"})
        else:
            df = pd.DataFrame(data, columns=["timestamp_ms", "open", "high", "low", "volatility"])

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"].astype(int), unit="ms", utc=True)
    df["currency"] = currency
    df = df[["currency", "timestamp_utc", "volatility"]].sort_values("timestamp_utc")
    return df.reset_index(drop=True)


def fetch_deribit_ohlcv(
    instrument: str,
    start_ts: int,
    end_ts: int,
    resolution: str = "60",
) -> pd.DataFrame:
    """
    Fetch OHLCV chart data from Deribit for any instrument (including options).
    resolution: '1', '3', '5', '10', '15', '30', '60', '120', '180', '360', '720', '1D'

    Returns DataFrame with timestamp_utc, open, high, low, close, volume.
    """
    resp = requests.get(
        f"{DERIBIT_BASE}/get_tradingview_chart_data",
        params={
            "instrument_name": instrument,
            "start_timestamp": start_ts * 1000,  # expects ms
            "end_timestamp": end_ts * 1000,
            "resolution": resolution,
        },
    )
    resp.raise_for_status()
    result = resp.json()["result"]

    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(result["ticks"], unit="ms", utc=True),
        "open": result["open"],
        "high": result["high"],
        "low": result["low"],
        "close": result["close"],
        "volume": result["volume"],
    })
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def save_deribit_dvol(currency: str, df: pd.DataFrame) -> Path:
    out = RAW_DIR / "deribit" / f"dvol_{currency.lower()}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load_deribit_dvol(currency: str = "BTC") -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / "deribit" / f"dvol_{currency.lower()}.parquet")


# ─────────────────────────────────────────────
# LUNARCRUSH — Sentiment Baseline
# ─────────────────────────────────────────────

LUNARCRUSH_BASE = "https://lunarcrush.com/api4/public"


def fetch_lunarcrush_sentiment(coin: str = "btc", api_key: str = None) -> pd.DataFrame:
    """
    Fetch LunarCrush Galaxy Score (social sentiment) for a crypto asset.
    Free tier has limited historical depth.

    Returns DataFrame with timestamp_utc, galaxy_score, sentiment, social_volume.
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.get(
        f"{LUNARCRUSH_BASE}/coins/{coin}/v1",
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()

    # LunarCrush v4 returns timeseries in data.timeSeries
    ts = data.get("data", {}).get("timeSeries", [])
    if not ts:
        return pd.DataFrame()

    df = pd.DataFrame(ts)
    df["timestamp_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)

    cols_keep = {
        "galaxy_score": "galaxy_score",
        "sentiment": "sentiment",
        "social_volume": "social_volume",
        "social_score": "social_score",
    }
    rename = {k: v for k, v in cols_keep.items() if k in df.columns}
    df = df.rename(columns=rename)

    out_cols = ["timestamp_utc"] + list(rename.values())
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].sort_values("timestamp_utc").reset_index(drop=True)


def save_lunarcrush(coin: str, df: pd.DataFrame) -> Path:
    out = RAW_DIR / "lunarcrush" / f"sentiment_{coin}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
