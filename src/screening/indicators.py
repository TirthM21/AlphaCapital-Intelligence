"""Technical indicators module for stock analysis.

This module provides implementations of common technical indicators used in
stock screening and analysis, including RSI, SMA, EMA, and volume analysis.
"""

import logging
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI).

    RSI is a momentum oscillator that measures the speed and magnitude of
    directional price changes. RSI oscillates between 0 and 100.

    Args:
        prices: Series of closing prices.
        period: Number of periods for RSI calculation (default: 14).

    Returns:
        Series of RSI values. Values range from 0 to 100.
        - RSI > 70: Overbought condition
        - RSI < 30: Oversold condition

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> rsi = calculate_rsi(prices, period=6)
        >>> print(rsi.iloc[-1])
    """
    if len(prices) < period + 1:
        logger.warning(f"Insufficient data for RSI calculation: {len(prices)} < {period + 1}")
        return pd.Series([np.nan] * len(prices), index=prices.index)

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate exponential moving averages
    avg_gains = gains.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gains / avg_losses.replace(0, np.nan)

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA).

    SMA is the arithmetic mean of prices over a specified period.

    Args:
        prices: Series of closing prices.
        period: Number of periods for SMA calculation.

    Returns:
        Series of SMA values.

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105])
        >>> sma = calculate_sma(prices, period=3)
        >>> print(sma.iloc[-1])
    """
    if len(prices) < period:
        logger.warning(f"Insufficient data for SMA calculation: {len(prices)} < {period}")
        return pd.Series([np.nan] * len(prices), index=prices.index)

    return prices.rolling(window=period, min_periods=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    EMA gives more weight to recent prices, making it more responsive to
    new information compared to SMA.

    Args:
        prices: Series of closing prices.
        period: Number of periods for EMA calculation.

    Returns:
        Series of EMA values.

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105])
        >>> ema = calculate_ema(prices, period=3)
        >>> print(ema.iloc[-1])
    """
    if len(prices) < period:
        logger.warning(f"Insufficient data for EMA calculation: {len(prices)} < {period}")
        return pd.Series([np.nan] * len(prices), index=prices.index)

    return prices.ewm(span=period, min_periods=period, adjust=False).mean()


def find_swing_lows(prices: pd.Series, window: int = 30) -> List[float]:
    """Find swing lows (local minimums) in price data.

    A swing low is identified when a price point is lower than prices within
    a specified window on both sides.

    Args:
        prices: Series of closing prices.
        window: Number of periods on each side to check (default: 30).

    Returns:
        List of price levels representing swing lows, sorted ascending.

    Example:
        >>> prices = pd.Series([100, 95, 90, 92, 95, 93, 88, 90, 95])
        >>> lows = find_swing_lows(prices, window=2)
        >>> print(lows)  # [88.0, 90.0]
    """
    if len(prices) < window * 2 + 1:
        logger.warning(f"Insufficient data for swing low detection: {len(prices)} < {window * 2 + 1}")
        return []

    swing_lows = []

    # Find local minimums
    for i in range(window, len(prices) - window):
        current_price = prices.iloc[i]

        # Check if current price is lower than all prices in the window
        left_window = prices.iloc[i - window:i]
        right_window = prices.iloc[i + 1:i + window + 1]

        if (current_price <= left_window.min()) and (current_price <= right_window.min()):
            swing_lows.append(float(current_price))

    # Remove duplicates and sort
    swing_lows = sorted(list(set(swing_lows)))

    logger.debug(f"Found {len(swing_lows)} swing lows")
    return swing_lows


def detect_volume_spike(
    volumes: pd.Series,
    current_volume: float,
    threshold: float = 1.5
) -> bool:
    """Detect if current volume is significantly higher than average.

    A volume spike indicates increased trading activity, which can signal
    important price movements or accumulation/distribution.

    Args:
        volumes: Series of historical volume data.
        current_volume: Current period's volume.
        threshold: Multiplier for average volume (default: 1.5 = 150% of average).

    Returns:
        True if current volume exceeds threshold * average volume, False otherwise.

    Example:
        >>> volumes = pd.Series([1000000, 1100000, 950000, 1050000])
        >>> is_spike = detect_volume_spike(volumes, 1600000, threshold=1.5)
        >>> print(is_spike)  # True
    """
    if len(volumes) < 20:
        logger.warning(f"Insufficient volume data: {len(volumes)} < 20")
        return False

    # Calculate average volume (last 20 periods)
    avg_volume = volumes.iloc[-20:].mean()

    if avg_volume == 0:
        return False

    # Check if current volume exceeds threshold
    is_spike = current_volume >= (avg_volume * threshold)

    if is_spike:
        logger.debug(f"Volume spike detected: {current_volume:.0f} vs avg {avg_volume:.0f}")

    return is_spike


def calculate_support_strength(
    prices: pd.Series,
    support_level: float,
    tolerance: float = 0.02
) -> int:
    """Calculate the strength of a support level.

    Support strength is measured by the number of times the price has
    touched and bounced off a support level.

    Args:
        prices: Series of closing prices.
        support_level: The support price level to test.
        tolerance: Percentage tolerance for considering a touch (default: 2%).

    Returns:
        Integer representing the number of times support was tested.

    Example:
        >>> prices = pd.Series([100, 95, 96, 94, 97, 95, 98])
        >>> strength = calculate_support_strength(prices, 95.0, tolerance=0.02)
        >>> print(strength)
    """
    if len(prices) == 0:
        return 0

    # Calculate price range for support level
    lower_bound = support_level * (1 - tolerance)
    upper_bound = support_level * (1 + tolerance)

    # Count touches
    touches = ((prices >= lower_bound) & (prices <= upper_bound)).sum()

    return int(touches)


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) and two outer bands
    (standard deviations away from the middle band).

    Args:
        prices: Series of closing prices.
        period: Number of periods for SMA (default: 20).
        num_std: Number of standard deviations for bands (default: 2.0).

    Returns:
        Tuple of (middle_band, upper_band, lower_band).

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> middle, upper, lower = calculate_bollinger_bands(prices, period=5)
        >>> print(f"Middle: {middle.iloc[-1]:.2f}")
    """
    if len(prices) < period:
        empty = pd.Series([np.nan] * len(prices), index=prices.index)
        return empty, empty, empty

    # Calculate middle band (SMA)
    middle_band = calculate_sma(prices, period)

    # Calculate standard deviation
    rolling_std = prices.rolling(window=period, min_periods=period).std()

    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)

    return middle_band, upper_band, lower_band


def calculate_keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    multiplier: float = 2.0,
    atr_period: int = 10
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Keltner Channels."""
    if len(close) < max(period, atr_period):
        num = len(close)
        empty = pd.Series([np.nan] * num, index=close.index)
        return empty, empty, empty

    middle_line = close.ewm(span=period, adjust=False).mean()
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    upper_channel = middle_line + (atr * multiplier)
    lower_channel = middle_line - (atr * multiplier)

    return middle_line, upper_channel, lower_channel


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages.

    Args:
        prices: Series of closing prices.
        fast_period: Fast EMA period (default: 12).
        slow_period: Slow EMA period (default: 26).
        signal_period: Signal line EMA period (default: 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
        >>> macd, signal, histogram = calculate_macd(prices)
        >>> print(f"MACD: {macd.iloc[-1]:.2f}")
    """
    if len(prices) < slow_period + signal_period:
        empty = pd.Series([np.nan] * len(prices), index=prices.index)
        return empty, empty, empty

    # Calculate EMAs
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line
    signal_line = calculate_ema(macd_line, signal_period)

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range of an
    asset price for that period.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        period: Number of periods for ATR calculation (default: 14).

    Returns:
        Series of ATR values.

    Example:
        >>> high = pd.Series([102, 104, 103])
        >>> low = pd.Series([98, 99, 100])
        >>> close = pd.Series([100, 102, 101])
        >>> atr = calculate_atr(high, low, close, period=2)
    """
    if len(high) < period + 1:
        return pd.Series([np.nan] * len(high), index=high.index)

    # Calculate True Range
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate ATR (SMA of True Range)
    atr = true_range.rolling(window=period, min_periods=period).mean()

    return atr


def calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate Money Flow Index (MFI).

    MFI is a momentum indicator that measures the inflow and outflow of money
    into an asset over a specific period of time.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        volume: Series of volumes.
        period: Number of periods for MFI calculation (default: 14).

    Returns:
        Series of MFI values (0-100).
    """
    if len(high) < period + 1:
        return pd.Series([np.nan] * len(high), index=high.index)

    # Typical Price
    typical_price = (high + low + close) / 3

    # Raw Money Flow
    money_flow = typical_price * volume

    # Positive and Negative Money Flow
    diff = typical_price.diff()
    pos_mf = money_flow.where(diff > 0, 0.0)
    neg_mf = money_flow.where(diff < 0, 0.0)

    # Moving Sums
    pos_mf_sum = pos_mf.rolling(window=period, min_periods=period).sum()
    neg_mf_sum = neg_mf.rolling(window=period, min_periods=period).sum()

    # Money Flow Ratio
    mfr = pos_mf_sum / neg_mf_sum.replace(0, np.nan)

    # MFI
    mfi = 100 - (100 / (1 + mfr))

    return mfi


def calculate_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20
) -> pd.Series:
    """Calculate Commodity Channel Index (CCI).

    CCI measures the current price level relative to an average price level
    over a given period of time.

    Args:
        high: Series of high prices.
        low: Series of low prices.
        close: Series of closing prices.
        period: Number of periods for CCI calculation (default: 20).

    Returns:
        Series of CCI values.
    """
    if len(high) < period:
        return pd.Series([np.nan] * len(high), index=high.index)

    # Typical Price
    tp = (high + low + close) / 3

    # SMA of Typical Price
    sma_tp = tp.rolling(window=period, min_periods=period).mean()

    # Mean Deviation
    def mean_deviation(x):
        return np.mean(np.abs(x - np.mean(x)))

    m_dev = tp.rolling(window=period, min_periods=period).apply(mean_deviation)

    # CCI
    cci = (tp - sma_tp) / (0.015 * m_dev)

    return cci


def calculate_ttm_squeeze(
    close: pd.Series,
    low: pd.Series,
    high: pd.Series,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_std: float = 1.5
) -> pd.Series:
    """Detect TTM Squeeze (Bollinger Bands vs Keltner Channels).

    TTM Squeeze occurs when Bollinger Bands are inside Keltner Channels.
    This typically precedes a significant breakout.

    Returns:
        Series of booleans (True if in squeeze).
    """
    if len(close) < bb_period:
        return pd.Series([False] * len(close), index=close.index)

    # Bollinger Bands
    sma = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    bb_upper = sma + bb_std * std
    bb_lower = sma - bb_std * std

    # Keltner Channels
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(kc_period).mean()
    kc_upper = sma + kc_std * atr
    kc_lower = sma - kc_std * atr

    # Squeeze Condition
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    return squeeze


def calculate_psar(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2
) -> pd.Series:
    """Calculate Parabolic SAR (Stop and Reverse).

    Args:
        high: Series of high prices.
        low: Series of low prices.
    """
    length = len(high)
    psar = pd.Series([np.nan] * length, index=high.index)

    if length < 2:
        return psar

    # Initial values
    bull = True
    af = af_start
    ep = high.iloc[0]
    psar.iloc[0] = low.iloc[0]

    for i in range(1, length):
        prev_psar = psar.iloc[i-1]
        
        if bull:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[max(0, i-2)])
            
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                psar.iloc[i] = ep
                af = af_start
                ep = low.iloc[i]
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[max(0, i-2)])
            
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                psar.iloc[i] = ep
                af = af_start
                ep = high.iloc[i]
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_step, af_max)
                    
    return psar


def detect_nr4_nr7(df: pd.DataFrame) -> str:
    """Detect Narrow Range 4 (NR4) and Narrow Range 7 (NR7) patterns.
    
    NR4: Day's range is the narrowest in the last 4 days.
    NR7: Day's range is the narrowest in the last 7 days.
    """
    if len(df) < 7:
        return "none"
        
    ranges = (df['High'] - df['Low']).iloc[-7:]
    curr_range = ranges.iloc[-1]
    
    if curr_range == ranges.min():
        return "NR7"
    elif curr_range == ranges.iloc[-4:].min():
        return "NR4"
    return "none"


def detect_higher_hl(df: pd.DataFrame, window: int = 5) -> str:
    """Detect if stock is making higher highs and higher lows."""
    if len(df) < window + 1:
        return "none"
        
    prev = df.iloc[-window-1:-1]
    curr = df.iloc[-1]
    
    if curr['High'] > prev['High'].max() and curr['Low'] > prev['Low'].min():
        return "higher_hl"
    elif curr['High'] < prev['High'].min() and curr['Low'] < prev['Low'].max():
        return "lower_hl"
    return "none"


def calculate_lorentzian(df: pd.DataFrame, window: int = 20) -> float:
    """A simplified 'Lorentzian Classifier' approximation.
    
    In a real Lorentzian classifier, we use K-Nearest Neighbors on Lorentzian distance.
    Here we'll use a weighted score of multiple distance-based momentum features
    to approximate the 'probability' of a move.
    """
    if len(df) < window:
        return 50.0
        
    # Feature 1: RSI distance from midpoint
    rsi = calculate_rsi(df['Close'], 14).iloc[-1]
    f1 = (rsi - 50) / 50.0 # -1 to 1
    
    # Feature 2: Price distance from EMA
    ema = calculate_ema(df['Close'], 20).iloc[-1]
    f2 = (df['Close'].iloc[-1] - ema) / ema # % distance
    
    # Feature 3: Volume distance from avg
    vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
    f3 = (df['Volume'].iloc[-1] - vol_avg) / vol_avg
    
    # Weighted ensemble (Simplified classification)
    score = 50 + (f1 * 20) + (f2 * 100) + (f3 * 10)
    return max(0, min(100, score))


def detect_vsa_signals(df: pd.DataFrame) -> List[str]:
    """Volume Spread Analysis for smart money flow detection."""
    if len(df) < 5: return []
    
    signals = []
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    spread = curr['High'] - curr['Low']
    avg_spread = (df['High'] - df['Low']).rolling(20).mean().iloc[-1]
    
    # 1. Stopping Volume (Support)
    if curr['Close'] < prev['Low'] and curr['Volume'] > avg_vol * 1.5 and curr['Close'] > (curr['Low'] + spread * 0.4):
        signals.append("Stopping Volume (Potential Bottom)")
        
    # 2. No Demand Bar
    if curr['Volume'] < df['Volume'].iloc[-3:-1].min() and spread < avg_spread * 0.8 and curr['Close'] < prev['Close']:
        signals.append("No Demand Bar (Weakness)")
        
    # 3. Effort vs Result (Bullish)
    if curr['Volume'] > avg_vol * 1.2 and spread < avg_spread * 0.7 and curr['Close'] > prev['Close']:
        signals.append("Effort without Result (Absorption)")
        
    return signals


def find_support_resistance(df: pd.DataFrame, window: int = 50) -> Dict[str, float]:
    """Find major support and resistance levels."""
    if len(df) < window: return {'support': 0.0, 'resistance': 0.0}
    
    recent = df.tail(window)
    res = recent['High'].max()
    sup = recent['Low'].min()
    
    # Check for SMA support
    sma_200 = df['Close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else 0
    sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else 0
    
    return {
        'resistance_52w': df['High'].tail(252).max() if len(df) >= 252 else res,
        'support_52w': df['Low'].tail(252).min() if len(df) >= 252 else sup,
        'sma_50': round(sma_50, 2),
        'sma_200': round(sma_200, 2)
    }


def predict_next_day(df: pd.DataFrame) -> Dict[str, any]:
    """AI-style trend forecasting for the next day's price movement."""
    if len(df) < 20: return {'prediction': 'Neutral', 'confidence': 0}
    
    # Factors
    momentum = calculate_rsi(df['Close'], 14).iloc[-1]
    trend = calculate_ema(df['Close'], 20).iloc[-1]
    vol_trend = df['Volume'].iloc[-1] > df['Volume'].rolling(5).mean().iloc[-1]
    
    score = 0
    if df['Close'].iloc[-1] > trend: score += 20
    if momentum > 50: score += 15
    if vol_trend: score += 10
    if df['Close'].iloc[-1] > df['Open'].iloc[-1]: score += 15 # Bullish bar
    
    # SMA Crossovers
    sma_20 = df['Close'].rolling(20).mean().iloc[-1]
    sma_9 = df['Close'].rolling(9).mean().iloc[-1]
    if sma_9 > sma_20: score += 25
    
    prediction = "Bullish" if score >= 65 else ("Bearish" if score <= 35 else "Neutral")
    
    return {
        'prediction': prediction,
        'score': score,
        'confidence': min(100, score if score > 50 else (100 - score))
    }


def calculate_aroon(high: pd.Series, low: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Calculate Aroon Indicator (Aroon Up and Aroon Down)."""
    if len(high) < period:
        empty = pd.Series([np.nan] * len(high), index=high.index)
        return empty, empty

    def days_since_max(x):
        return period - (np.argmax(x) + 1)

    def days_since_min(x):
        return period - (np.argmin(x) + 1)

    aroon_up = high.rolling(window=period + 1).apply(days_since_max)
    aroon_down = low.rolling(window=period + 1).apply(days_since_min)

    aroon_up = (period - aroon_up) / period * 100
    aroon_down = (period - aroon_down) / period * 100

    return aroon_up, aroon_down


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate basic Fibonacci retracement levels.
    
    Levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
    """
    diff = high - low
    return {
        '0': round(low, 2),
        '0.236': round(low + 0.236 * diff, 2),
        '0.382': round(low + 0.382 * diff, 2),
        '0.5': round(low + 0.5 * diff, 2),
        '0.618': round(low + 0.618 * diff, 2),
        '0.786': round(low + 0.786 * diff, 2),
        '1': round(high, 2)
    }


def detect_candlestick_patterns(df: pd.DataFrame) -> List[str]:
    """Detect common candlestick patterns in the latest data."""
    if len(df) < 5:
        return []

    signals = []
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    curr_body = abs(curr['Close'] - curr['Open'])
    prev_body = abs(prev['Close'] - prev['Open'])
    
    curr_range = curr['High'] - curr['Low']
    curr_color = "green" if curr['Close'] > curr['Open'] else "red"
    prev_color = "green" if prev['Close'] > prev['Open'] else "red"

    # 1. Bullish Engulfing
    if prev_color == "red" and curr_color == "green" and curr['Close'] > prev['Open'] and curr['Open'] < prev['Close']:
        signals.append("Bullish Engulfing")

    # 2. Bearish Engulfing
    if prev_color == "green" and curr_color == "red" and curr['Close'] < prev['Open'] and curr['Open'] > prev['Close']:
        signals.append("Bearish Engulfing")

    # 3. Hammer (Bullish)
    lower_wick = min(curr['Open'], curr['Close']) - curr['Low']
    upper_wick = curr['High'] - max(curr['Open'], curr['Close'])
    if lower_wick > curr_body * 2 and upper_wick < curr_body * 0.5:
        signals.append("Hammer (Potential Reverse)")

    # 4. Shooting Star (Bearish)
    if upper_wick > curr_body * 2 and lower_wick < curr_body * 0.5:
        signals.append("Shooting Star")

    # 5. Morning Star (Bullish - 3 day pattern)
    if prev2['Close'] < prev2['Open'] and prev_body < abs(prev2['Close']-prev2['Open'])*0.5 and curr['Close'] > curr['Open'] and curr['Close'] > (prev2['Open'] + prev2['Close'])/2:
       signals.append("Morning Star")

    return signals
