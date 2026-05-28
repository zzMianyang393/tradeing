"""本地数据存储模块 - SQLite存储K线数据"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger


class DataStorage:
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_klines_symbol_tf_ts
                ON klines(symbol, timeframe, timestamp)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    size REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    open_time TEXT NOT NULL,
                    close_time TEXT NOT NULL,
                    close_reason TEXT NOT NULL
                )
            """)
            conn.commit()
        logger.debug(f"数据库初始化完成: {self.db_path}")

    def save_trade(self, record) -> None:
        """保存一笔交易记录到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trade_history
                (symbol, direction, entry_price, exit_price, size, leverage,
                 pnl, pnl_pct, open_time, close_time, close_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.symbol, record.direction,
                record.entry_price, record.exit_price,
                record.size, record.leverage,
                record.pnl, record.pnl_pct,
                record.open_time.isoformat() if hasattr(record.open_time, 'isoformat') else str(record.open_time),
                record.close_time.isoformat() if hasattr(record.close_time, 'isoformat') else str(record.close_time),
                record.close_reason,
            ))
            conn.commit()
        logger.debug(f"交易记录已保存: {record.symbol} {record.direction} pnl={record.pnl:.4f}U")

    def load_trades(self) -> list:
        """加载所有历史交易记录"""
        from execution.account import TradeRecord
        from datetime import datetime as dt
        trades = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM trade_history ORDER BY id ASC")
            for row in cursor.fetchall():
                trades.append(TradeRecord(
                    symbol=row[1],
                    direction=row[2],
                    entry_price=row[3],
                    exit_price=row[4],
                    size=row[5],
                    leverage=row[6],
                    pnl=row[7],
                    pnl_pct=row[8],
                    open_time=dt.fromisoformat(row[9]),
                    close_time=dt.fromisoformat(row[10]),
                    close_reason=row[11],
                ))
        return trades

    def save_klines(self, symbol: str, timeframe: str, df: pd.DataFrame):
        if df.empty:
            return

        records = []
        for _, row in df.iterrows():
            records.append((
                symbol,
                timeframe,
                int(row["timestamp"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            ))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO klines
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()

        logger.debug(f"保存 {symbol} {timeframe} K线 {len(records)} 条")

    def load_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM klines WHERE symbol = ? AND timeframe = ?"
        params = [symbol, timeframe]

        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp ASC"

        if limit is not None:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

        return df

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT MAX(timestamp) FROM klines WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            result = cursor.fetchone()
            return result[0] if result[0] is not None else None

    def get_kline_count(self, symbol: str, timeframe: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM klines WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            return cursor.fetchone()[0]

    def cleanup_old_data(self, keep_days: int = 90):
        cutoff = int((datetime.utcnow() - timedelta(days=keep_days)).timestamp() * 1000)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM klines WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
        if deleted > 0:
            logger.info(f"清理了 {deleted} 条过期数据（{keep_days}天前）")
