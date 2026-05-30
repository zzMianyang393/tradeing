"""生成多时间框架数据 - 从15m重采样为4h和1d"""

import pandas as pd
import numpy as np
from pathlib import Path


def resample(df_15m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """将15m数据重采样为目标时间框架"""
    df = df_15m.copy()

    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        df.index = pd.to_datetime(df.iloc[:, 0])

    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # 只聚合存在的列
    cols_to_use = [c for c in agg_dict.keys() if c in df.columns]
    df_resampled = df[cols_to_use].resample(target_tf).agg(
        {c: agg_dict[c] for c in cols_to_use}
    ).dropna()

    df_resampled = df_resampled.reset_index()
    # 重命名index列为timestamp（毫秒）
    if "index" in df_resampled.columns:
        df_resampled.rename(columns={"index": "timestamp"}, inplace=True)
    elif "timestamp" not in df_resampled.columns:
        df_resampled.rename(columns={df_resampled.columns[0]: "timestamp"}, inplace=True)

    # 转为毫秒时间戳
    if df_resampled["timestamp"].dtype != "int64":
        df_resampled["timestamp"] = df_resampled["timestamp"].astype(np.int64)

    return df_resampled


def main():
    data_dir = Path("data")

    for symbol in ["BTC", "ETH"]:
        src = data_dir / f"{symbol}_15m.csv"
        if not src.exists():
            print(f"跳过 {symbol}: {src} 不存在")
            continue

        df_15m = pd.read_csv(src)
        print(f"{symbol} 15m: {len(df_15m)} 根K线")

        for tf, label in [("4h", "4h"), ("1D", "1d")]:
            df_tf = resample(df_15m, tf)
            out_path = data_dir / f"{symbol}_{label}.csv"
            df_tf.to_csv(out_path, index=False)
            print(f"  {label}: {len(df_tf)} 根K线 -> {out_path}")

    print("\n完成！")


if __name__ == "__main__":
    main()
