
"""Piped Scanners for composite signal detection.

Combines multiple filters into sophisticated trading setups like those in PKScreener.
"""

from typing import Dict
import pandas as pd
from .extended_signals import score_extended_signals


def _has_signal(signals, pattern: str) -> bool:
    return any(pattern in signal for signal in signals)


def _count_signals(signals, patterns) -> int:
    return sum(_has_signal(signals, pattern) for pattern in patterns)


def run_piped_scan(ticker: str, df: pd.DataFrame, pipe_type: int) -> Dict:
    """Run a specific piped scanner logic on a stock."""
    
    # Get base technical signals first
    base = score_extended_signals(ticker, df)
    signals = base.get('signals', [])
    
    # Define PKScreener-style pipes
    setup_found = False
    pipe_name = ""
    
    if pipe_type == 1: # Volume + Momentum + Breakout
        pipe_name = "High Volume Momentum Breakout"
        if (
            _has_signal(signals, "Volume Breakout")
            and _count_signals(signals, [
                "Bullish 9/21 EMA Cross",
                "Bullish MACD Cross",
                "MACD Momentum Improving",
                "Upper Keltner Breakout",
            ]) > 0
            and _count_signals(signals, [
                "52-Week High Breakout",
                "Higher High / Higher Low Trend",
                "Squeeze Breakout",
            ]) > 0
            and base.get('bullish_score', 0) > base.get('bearish_score', 0)
            and not _has_signal(signals, "52-Week Low")
            and not _has_signal(signals, "Lower High / Lower Low Trend")
        ):
            setup_found = True
            
    elif pipe_type == 2: # VCP + Chart Pattern + MA Support
        pipe_name = "VCP & Chart Pattern Setup"
        if (
            _count_signals(signals, [
                "Narrow Range",
                "Inside Bar",
                "Keltner Band Compression",
                "TTM Squeeze",
            ]) > 0
            and _count_signals(signals, [
                "Potential Support Bounce Zone",
                "Higher High / Higher Low Trend",
                "Near Fibonacci Level",
            ]) > 0
            and base.get('bullish_score', 0) >= base.get('bearish_score', 0)
            and not _has_signal(signals, "Lower High / Lower Low Trend")
        ):
            setup_found = True
                
    elif pipe_type == 3: # Golden Cross + Bullish Lorentzian
        pipe_name = "Institutional Trend Setup"
        if (
            _count_signals(signals, ["Golden Cross", "Higher High / Higher Low Trend"]) > 0
            and base.get('lorentzian', 0) >= 65
            and _count_signals(signals, ["Near 52-Week High Case", "52-Week High Breakout", "Volume Breakout"]) > 0
            and base.get('bullish_score', 0) > base.get('bearish_score', 0)
            and not _has_signal(signals, "Lower High / Lower Low Trend")
        ):
            setup_found = True
            
    elif pipe_type == 4: # RSI Divergence + Candlestick Reversal
        pipe_name = "Mean Reversion Pivot"
        if (
            _count_signals(signals, ["Bullish RSI Divergence", "RSI Oversold"]) > 0
            and _count_signals(signals, [
                "Candlestick: Bullish Engulfing",
                "Candlestick: Hammer",
                "Candlestick: Morning Star",
            ]) > 0
            and _count_signals(signals, [
                "Potential Support Bounce Zone",
                "CCI Low",
                "MFI Low",
            ]) > 0
            and base.get('bullish_score', 0) >= base.get('bearish_score', 0)
            and not _has_signal(signals, "Lower High / Lower Low Trend")
        ):
            setup_found = True

    base['pipe_name'] = pipe_name
    base['setup_found'] = setup_found
    if setup_found:
        base['signals'] = list(base.get('signals', []))
        base['signals'].append(f"PIPED: {pipe_name}")
        base['prediction_score'] += 10
        
    return base
