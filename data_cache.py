"""SQLite-based historical data cache to reduce IBKR data requests."""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CLIENT_ID, CURRENCY, EXCHANGE, HOST, PORT, SYMBOL

CACHE_TTL = dt.timedelta(hours=1)
DB_PATH = Path("data_cache.sqlite")
logger = logging.getLogger(__name__)


def _normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized OHLCV frame with lowercase columns and a time column.

    This helper attempts to be resilient to data from external sources that may
    arrive with differently cased column names or with a DateTimeIndex. When
    required columns are missing, an empty DataFrame is returned to avoid
    downstream KeyErrors during resampling.
    """

    if df.empty:
        return df

    normalized = df.copy()
    normalized = normalized.rename(columns={col: str(col).lower() for col in normalized.columns})

    if "time" not in normalized.columns:
        normalized = normalized.reset_index()
        normalized = normalized.rename(columns={col: str(col).lower() for col in normalized.columns})
        if "time" not in normalized.columns and "index" in normalized.columns:
            normalized = normalized.rename(columns={"index": "time"})

    required_cols = {"time", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(normalized.columns)
    if missing:
        logger.warning("Dropping dataframe missing required OHLCV columns: %s", sorted(missing))
        return pd.DataFrame(columns=sorted(required_cols))

    normalized["time"] = pd.to_datetime(normalized["time"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    normalized = normalized.dropna(subset=["time"])

    return normalized[["time", "open", "high", "low", "close", "volume"]]

def _to_naive_utc(ts: dt.datetime) -> dt.datetime:
    """Normalize timestamps to naive UTC for consistent storage/processing."""
    if ts.tzinfo:
        return ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return ts.replace(tzinfo=None)


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bars (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            currency TEXT NOT NULL,
            timeframe INTEGER NOT NULL,
            time TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY(symbol, exchange, currency, timeframe, time)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_state (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            currency TEXT NOT NULL,
            timeframe INTEGER NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY(symbol, exchange, currency, timeframe)
        )
        """
    )


def _cache_key(timeframe_minutes: int, symbol: str, exchange: str, currency: str) -> tuple[str, str, str, int]:
    return (symbol, exchange, currency, timeframe_minutes)


def _is_fresh(fetched_at: str) -> bool:
    try:
        fetched_time = dt.datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    return dt.datetime.utcnow() - fetched_time <= CACHE_TTL


def _read_cached_bars(timeframe_minutes: int, symbol: str, exchange: str, currency: str) -> Optional[pd.DataFrame]:
    if not DB_PATH.exists():
        return None

    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn)
        key = _cache_key(timeframe_minutes, symbol, exchange, currency)
        row = conn.execute(
            "SELECT fetched_at FROM cache_state WHERE symbol=? AND exchange=? AND currency=? AND timeframe=?",
            key,
        ).fetchone()
        if not row or not _is_fresh(row[0]):
            return None

        df = pd.read_sql_query(
            """
            SELECT time, open, high, low, close, volume
            FROM bars
            WHERE symbol=? AND exchange=? AND currency=? AND timeframe=?
            ORDER BY time
            """,
            conn,
            params=key,
        )
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
        return df


def _write_cached_bars(timeframe_minutes: int, df: pd.DataFrame, symbol: str, exchange: str, currency: str) -> None:
    if df.empty:
        return

    normalized = _normalize_ohlcv_frame(df)
    if normalized.empty:
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [
        (
            symbol,
            exchange,
            currency,
            timeframe_minutes,
            pd.to_datetime(row.time, utc=True, errors="coerce").tz_convert("UTC").tz_localize(None).isoformat(),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        )
        for row in normalized.itertuples()
        if getattr(row, "time", None) is not None
    ]
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn)
        with conn:
            conn.execute(
                "DELETE FROM bars WHERE symbol=? AND exchange=? AND currency=? AND timeframe=?",
                _cache_key(timeframe_minutes, symbol, exchange, currency),
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO bars
                (symbol, exchange, currency, timeframe, time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_state (symbol, exchange, currency, timeframe, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (*_cache_key(timeframe_minutes, symbol, exchange, currency), dt.datetime.utcnow().isoformat()),
            )


def _fetch_hourly_from_ib(lookback_days: int, symbol: str, exchange: str, currency: str) -> pd.DataFrame:
    """Lazy-import ib_insync to avoid event loop creation when unused."""
    try:
        from ib_insync import Contract, IB, Stock
    except Exception as exc:  # pragma: no cover - defensive import path
        raise RuntimeError("ib_insync is required for IBKR data fetching") from exc

    ib = IB()
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID + 200, timeout=5)
        contract: Contract = Stock(symbol, exchange, currency)
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{lookback_days} D",
            barSizeSetting="1 hour",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        rows = []
        for b in bars:
            raw_ts = b.date if isinstance(b.date, dt.datetime) else dt.datetime.strptime(b.date, "%Y%m%d %H:%M:%S")
            ts = _to_naive_utc(raw_ts)
            rows.append(
                {
                    "time": ts,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
            )
        return pd.DataFrame(rows)
    finally:
        ib.disconnect()


def _fetch_hourly_from_yfinance(lookback_days: int, symbol: str) -> pd.DataFrame:
    """Fallback hourly fetch using yfinance when IBKR is unavailable."""
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("yfinance not available for fallback data: %s", exc)
        return pd.DataFrame()

    period_days = min(lookback_days, 730)  # yfinance caps at ~730 days for 60m interval
    try:
        raw = yf.download(symbol, period=f"{period_days}d", interval="60m", progress=False)
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("yfinance download failed: %s", exc)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index().rename(
        columns={"Datetime": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    raw["time"] = pd.to_datetime(raw["time"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    return raw[["time", "open", "high", "low", "close", "volume"]]


def _hourly_dataframe(lookback_days: int, symbol: str, exchange: str, currency: str) -> pd.DataFrame:
    base_df = _read_cached_bars(60, symbol, exchange, currency)
    if base_df is None:
        try:
            base_df = _fetch_hourly_from_ib(lookback_days, symbol, exchange, currency)
        except RuntimeError as exc:
            logger.warning("Unable to fetch hourly data from IBKR: %s", exc)
            base_df = _fetch_hourly_from_yfinance(lookback_days, symbol)
        if not base_df.empty:
            _write_cached_bars(60, base_df, symbol, exchange, currency)

    if base_df is None:
        return pd.DataFrame()

    if not base_df.empty:
        cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).tz_localize(None)
        base_df = base_df[base_df["time"] >= cutoff]
    return base_df


def _resample_dataframe(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    normalized = _normalize_ohlcv_frame(df)
    if normalized.empty:
        return normalized

    indexed = normalized.sort_values("time").set_index("time")
    rule = f"{timeframe_minutes}T"
    agg = indexed.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return agg


def get_resampled_bars(
    timeframe_minutes: int,
    lookback_days: int,
    symbol: str = SYMBOL,
    exchange: str = EXCHANGE,
    currency: str = CURRENCY,
) -> pd.DataFrame:
    """Return cached/resampled bars for the requested timeframe."""
    hourly_df = _hourly_dataframe(lookback_days, symbol, exchange, currency)
    if timeframe_minutes == 60:
        return hourly_df
    return _resample_dataframe(hourly_df, timeframe_minutes)


def fetch_historical_dataframe(
    timeframe_minutes: int,
    lookback_days: int,
    *,
    symbol: str = SYMBOL,
    exchange: str = EXCHANGE,
    currency: str = CURRENCY,
) -> pd.DataFrame:
    """Public helper to fetch/resample bars for consumers like backtests and dashboards."""
    return get_resampled_bars(timeframe_minutes, lookback_days, symbol=symbol, exchange=exchange, currency=currency)
