"""SQLite-based historical data cache to reduce IBKR data requests."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CLIENT_ID, CURRENCY, EXCHANGE, HOST, PORT, SYMBOL

CACHE_TTL = dt.timedelta(hours=1)
DB_PATH = Path("data_cache.sqlite")

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


def _cache_key(timeframe_minutes: int) -> tuple[str, str, str, int]:
    return (SYMBOL, EXCHANGE, CURRENCY, timeframe_minutes)


def _is_fresh(fetched_at: str) -> bool:
    try:
        fetched_time = dt.datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    return dt.datetime.utcnow() - fetched_time <= CACHE_TTL


def _read_cached_bars(timeframe_minutes: int) -> Optional[pd.DataFrame]:
    if not DB_PATH.exists():
        return None

    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn)
        key = _cache_key(timeframe_minutes)
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


def _write_cached_bars(timeframe_minutes: int, df: pd.DataFrame) -> None:
    if df.empty:
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [
        (
            SYMBOL,
            EXCHANGE,
            CURRENCY,
            timeframe_minutes,
            row.time.isoformat(),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        )
        for row in df.itertuples()
    ]
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_tables(conn)
        with conn:
            conn.execute(
                "DELETE FROM bars WHERE symbol=? AND exchange=? AND currency=? AND timeframe=?",
                _cache_key(timeframe_minutes),
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
                (*_cache_key(timeframe_minutes), dt.datetime.utcnow().isoformat()),
            )


def _fetch_hourly_from_ib(lookback_days: int) -> pd.DataFrame:
    """Lazy-import ib_insync to avoid event loop creation when unused."""
    try:
        from ib_insync import Contract, IB, Stock
    except Exception as exc:  # pragma: no cover - defensive import path
        raise RuntimeError("ib_insync is required for IBKR data fetching") from exc

    ib = IB()
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID + 200, timeout=5)
        contract: Contract = Stock(SYMBOL, EXCHANGE, CURRENCY)
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


def _hourly_dataframe(lookback_days: int) -> pd.DataFrame:
    base_df = _read_cached_bars(60)
    if base_df is None:
        base_df = _fetch_hourly_from_ib(lookback_days)
        if not base_df.empty:
            _write_cached_bars(60, base_df)

    if base_df is None:
        return pd.DataFrame()

    if not base_df.empty:
        cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).tz_localize(None)
        base_df = base_df[base_df["time"] >= cutoff]
    return base_df


def _resample_dataframe(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    indexed = df.copy()
    indexed["time"] = pd.to_datetime(indexed["time"], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    indexed = indexed.sort_values("time").set_index("time")
    rule = f"{timeframe_minutes}T"
    agg = indexed.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return agg


def get_resampled_bars(timeframe_minutes: int, lookback_days: int) -> pd.DataFrame:
    """Return cached/resampled bars for the requested timeframe."""
    hourly_df = _hourly_dataframe(lookback_days)
    if timeframe_minutes == 60:
        return hourly_df
    return _resample_dataframe(hourly_df, timeframe_minutes)


def fetch_historical_dataframe(timeframe_minutes: int, lookback_days: int) -> pd.DataFrame:
    """Public helper to fetch/resample bars for consumers like backtests and dashboards."""
    return get_resampled_bars(timeframe_minutes, lookback_days)
