"""
pandas_ta shim — 纯 pandas/numpy 实现的 ta 模块
替代已从 PyPI 下架的 pandas-ta，提供完全兼容的 API
"""
import numpy as np
import pandas as pd


def ema(close, length=9):
    """Exponential Moving Average"""
    return close.ewm(span=length, adjust=False).mean()


def sma(close, length=20):
    """Simple Moving Average"""
    return close.rolling(window=length).mean()


def rsi(close, length=14):
    """Relative Strength Index"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(close, fast=12, slow=26, signal=9):
    """MACD — returns DataFrame with columns: MACD, histogram, signal"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        f"MACD_{fast}_{slow}_{signal}": macd_line,
        f"MACDh_{fast}_{slow}_{signal}": histogram,
        f"MACDs_{fast}_{slow}_{signal}": signal_line,
    })


def bbands(close, length=20, std=2):
    """Bollinger Bands — returns DataFrame with columns: lower, middle, upper"""
    middle = sma(close, length)
    rolling_std = close.rolling(window=length).std()
    upper = middle + std * rolling_std
    lower = middle - std * rolling_std
    return pd.DataFrame({
        f"BBL_{length}_{std}": lower,
        f"BBM_{length}_{std}": middle,
        f"BBU_{length}_{std}": upper,
    })


def atr(high, low, close, length=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()


def adx(high, low, close, length=14):
    """Average Directional Index — returns DataFrame with columns: ADX, DMP, DMN"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_val = atr(high, low, close, length)

    plus_di = 100 * (plus_dm.ewm(alpha=1/length, min_periods=length, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, min_periods=length, adjust=False).mean() / atr_val)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

    return pd.DataFrame({
        f"ADX_{length}": adx_val,
        f"DMP_{length}": plus_di,
        f"DMN_{length}": minus_di,
    })


def obv(close, volume):
    """On Balance Volume"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()


def stochrsi(close, length=14, rsi_length=14, smooth_k=3, smooth_d=3):
    """Stochastic RSI — returns DataFrame with columns: K, D"""
    rsi_val = rsi(close, rsi_length)
    rsi_min = rsi_val.rolling(window=length).min()
    rsi_max = rsi_val.rolling(window=length).max()
    stoch_rsi = (rsi_val - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k = stoch_rsi.rolling(window=smooth_k).mean() * 100
    d = k.rolling(window=smooth_d).mean()
    return pd.DataFrame({
        "STOCHRSIk": k,
        "STOCHRSId": d,
    })
