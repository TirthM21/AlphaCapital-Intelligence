"""Extended screening signals for diverse trading strategies.

This module provides detection for a wide range of trading signals including
momentum, trend reversals, crossovers, and price patterns.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    calculate_macd,
    calculate_mfi,
    calculate_cci,
    calculate_ttm_squeeze,
    calculate_psar,
    detect_volume_spike,
    calculate_atr,
    detect_nr4_nr7,
    detect_higher_hl,
    calculate_lorentzian,
    detect_vsa_signals,
    find_support_resistance,
    predict_next_day,
    calculate_aroon,
    calculate_fibonacci_levels,
    detect_candlestick_patterns,
    calculate_keltner_channels
)

logger = logging.getLogger(__name__)

def detect_crossover(series_fast: pd.Series, series_slow: pd.Series) -> str:
    """Detect if fast series crossed above or below the slow series."""
    if len(series_fast) < 2 or len(series_slow) < 2:
        return "none"
    
    if series_fast.iloc[-2] <= series_slow.iloc[-2] and series_fast.iloc[-1] > series_slow.iloc[-1]:
        return "bullish"
    elif series_fast.iloc[-2] >= series_slow.iloc[-2] and series_fast.iloc[-1] < series_slow.iloc[-1]:
        return "bearish"
    return "none"

def detect_inside_bar(df: pd.DataFrame) -> bool:
    """Detect if the current bar is inside the previous bar."""
    if len(df) < 2:
        return False
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    return (curr['High'] < prev['High']) and (curr['Low'] > prev['Low'])

def detect_divergence(prices: pd.Series, indicator: pd.Series, window: int = 40) -> str:
    """Detect bullish or bearish divergence."""
    if len(prices) < window or len(indicator) < window:
        return "none"
        
    # Look for last 2 lows in price
    price_lows = []
    ind_lows = []
    
    # Simple check for the last 40 days - find 2 swing lows
    for i in range(5, window - 5):
        if prices.iloc[-i] < prices.iloc[-i+5] and prices.iloc[-i] < prices.iloc[-i-5]:
            price_lows.append(-i)
            ind_lows.append(indicator.iloc[-i])
            if len(price_lows) >= 2: break
            
    if len(price_lows) >= 2:
        # Bullish: Lower low in price, higher low in indicator
        if prices.iloc[price_lows[0]] < prices.iloc[price_lows[1]] and ind_lows[0] > ind_lows[1]:
            return "bullish"
            
    # Bearish: Higher high in price, lower high in indicator
    price_highs = []
    ind_highs = []
    for i in range(5, window - 5):
        if prices.iloc[-i] > prices.iloc[-i+5] and prices.iloc[-i] > prices.iloc[-i-5]:
            price_highs.append(-i)
            ind_highs.append(indicator.iloc[-i])
            if len(price_highs) >= 2: break
            
    if len(price_highs) >= 2:
        if prices.iloc[price_highs[0]] > prices.iloc[price_highs[1]] and ind_highs[0] < ind_highs[1]:
            return "bearish"
            
    return "none"

def score_extended_signals(ticker: str, df: pd.DataFrame) -> Dict:
    """Generate a comprehensive set of signals for a stock."""
    if len(df) < 200:
        return {'ticker': ticker, 'error': 'Insufficient data'}
    
    signals = []
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']
    
    # 1. Moving Averages
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    ema_9 = calculate_ema(close, 9)
    ema_21 = calculate_ema(close, 21)
    
    # Golden / Death Cross
    ma_cross = detect_crossover(sma_50, sma_200)
    if ma_cross == "bullish":
        signals.append("Golden Cross")
    elif ma_cross == "bearish":
        signals.append("Death Cross")
        
    # EMA Cross (Short term)
    ema_cross = detect_crossover(ema_9, ema_21)
    if ema_cross == "bullish":
        signals.append("Bullish 9/21 EMA Cross")
    elif ema_cross == "bearish":
        signals.append("Bearish 9/21 EMA Cross")
    
    # 2. Momentum Indicators
    rsi = calculate_rsi(close, 14)
    mfi = calculate_mfi(high, low, close, vol, 14)
    cci = calculate_cci(high, low, close, 20)
    
    latest_rsi = rsi.iloc[-1]
    latest_mfi = mfi.iloc[-1]
    latest_cci = cci.iloc[-1]
    
    if latest_rsi > 70: signals.append("RSI Overbought")
    elif latest_rsi < 30: signals.append("RSI Oversold")
    
    # RSI Divergence
    rsi_div = detect_divergence(close, rsi)
    if rsi_div == "bullish": signals.append("Bullish RSI Divergence")
    elif rsi_div == "bearish": signals.append("Bearish RSI Divergence")
    
    if latest_mfi > 80: signals.append("MFI High")
    elif latest_mfi < 20: signals.append("MFI Low")
    
    if latest_cci > 100: signals.append("CCI High")
    elif latest_cci < -100: signals.append("CCI Low")
    
    # 3. Volume
    if detect_volume_spike(vol, vol.iloc[-1], threshold=2.0):
        signals.append("Volume Breakout (>2x)")
        
    # 4. Price Patterns
    if detect_inside_bar(df):
        signals.append("Inside Bar")
        
    week_52_high = df['High'].rolling(252).max().iloc[-1]
    week_52_low = df['Low'].rolling(252).min().iloc[-1]
    
    if close.iloc[-1] >= week_52_high:
        signals.append("52-Week High Breakout")
    elif close.iloc[-1] <= week_52_low:
        signals.append("52-Week Low")
        
    # 5. Volatility & Trend
    psar = calculate_psar(high, low)
    if psar.iloc[-1] < close.iloc[-1] and psar.iloc[-2] >= close.iloc[-2]:
        signals.append("PSAR Trend Reversal (Bullish)")
    elif psar.iloc[-1] > close.iloc[-1] and psar.iloc[-2] <= close.iloc[-2]:
        signals.append("PSAR Trend Reversal (Bearish)")
    # 13. TTM Squeeze Detection
    squeeze_series = calculate_ttm_squeeze(close, df['Low'], df['High'])
    squeeze = squeeze_series.iloc[-1]
    if squeeze:
        signals.append("TTM Squeeze (Potential Breakout)")
        
    # 14. Keltner Channels
    k_mid, k_upper, k_lower = calculate_keltner_channels(df['High'], df['Low'], close)
    curr_price = close.iloc[-1]
    
    if curr_price > k_upper.iloc[-1]:
        signals.append("Upper Keltner Breakout (Bullish Momentum)")
    elif curr_price < k_lower.iloc[-1]:
        signals.append("Lower Keltner Rejection (Oversold)")
    
    # Check for Keltner squeeze (narrow bands)
    k_range = (k_upper - k_lower) / k_mid
    if k_range.iloc[-1] < k_range.tail(20).mean() * 0.8:
        signals.append("Keltner Band Compression (Volatility Squeeze)")
    elif squeeze_series.iloc[-2] and not squeeze_series.iloc[-1]: # Use squeeze_series for historical check
        signals.append("Squeeze Breakout")
        
    # ATR Expansion
    atr = calculate_atr(high, low, close, 14)
    if atr.iloc[-1] > atr.iloc[-2] * 1.5:
        signals.append("ATR Expansion (Volatility Jump)")
        
    # 6. Advanced Patterns
    nr = detect_nr4_nr7(df)
    if nr != "none":
        signals.append(f"Narrow Range ({nr}) - Consolidation")
        
    hl = detect_higher_hl(df)
    if hl == "higher_hl":
        signals.append("Higher High / Higher Low Trend")
    elif hl == "lower_hl":
        signals.append("Lower High / Lower Low Trend")
        
    lorentz_score = calculate_lorentzian(df)
    if lorentz_score >= 70:
        signals.append(f"Lorentzian Classifier (Bullish: {lorentz_score:.1f})")
    elif lorentz_score <= 30:
        signals.append(f"Lorentzian Classifier (Bearish: {lorentz_score:.1f})")
        
    # 7. VSA Analysis
    vsa = detect_vsa_signals(df)
    for v in vsa:
        signals.append(f"VSA: {v}")
        
    # 8. Support / Resistance
    sr = find_support_resistance(df)
    if close.iloc[-1] > sr['resistance_52w'] * 0.98:
        signals.append("Near 52-Week High Case")
    if close.iloc[-1] < sr['support_52w'] * 1.02:
        signals.append("Potential Support Bounce Zone")
        
    # 8b. Fibonacci Awareness
    fib = calculate_fibonacci_levels(sr['resistance_52w'], sr['support_52w'])
    curr_price = close.iloc[-1]
    for lvl, price in fib.items():
        if abs(curr_price - price) / price < 0.01:
            signals.append(f"Near Fibonacci Level: {lvl}")
            
    # 9. Aroon Crossover
    aroon_up, aroon_down = calculate_aroon(high, low)
    if aroon_up.iloc[-1] > 70 and aroon_up.iloc[-2] <= 70:
        signals.append("Aroon Bullish Crossover")
        
    # 10. Candlestick Patterns
    candles = detect_candlestick_patterns(df)
    for c in candles:
        signals.append(f"Candlestick: {c}")

    # 11. Next Day AI Prediction
    pred_data = predict_next_day(df)
    prediction_text = f"Next Day Prediction: {pred_data['prediction']} ({pred_data['confidence']}% confidence)"
        
    # 10. "AI" Style Prediction Score (Weighted probability)
    prediction_score = pred_data['score']
    # Adjust based on other signals
    if signals and any("Bullish" in s or "higher_hl" in s or "Golden" in s or "Lorentzian" in s or "VCP" in s for s in signals): 
        prediction_score += 15
    
    return {
        'ticker': ticker,
        'signals': signals,
        'price': round(close.iloc[-1], 2),
        'rsi': round(latest_rsi, 1),
        'volume_ratio': round(vol.iloc[-1] / vol.iloc[-50:-1].mean(), 2) if len(vol) > 50 else 1.0,
        'prediction_score': min(100, prediction_score),
        'lorentzian': round(lorentz_score, 1),
        'next_day_forecast': prediction_text,
        'levels': sr,
        'fib_levels': fib
    }
