
"""Piped Scanners for composite signal detection.

Combines multiple filters into sophisticated trading setups like those in PKScreener.
"""

from typing import List, Dict
import pandas as pd
from .extended_signals import score_extended_signals

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
        if any("Volume Breakout" in s for s in signals) and \
           any("RSI" in s and "High" not in s for s in signals) and \
           any("52-Week High Breakout" in s for s in signals):
            setup_found = True
            
    elif pipe_type == 2: # VCP + Chart Pattern + MA Support
        pipe_name = "VCP & Chart Pattern Setup"
        if any("VCP" in s for s in signals) or any("Narrow Range" in s for s in signals):
            if any("Support" in s for s in signals):
                setup_found = True
                
    elif pipe_type == 3: # Golden Cross + Bullish Lorentzian
        pipe_name = "Institutional Trend Setup"
        if "Golden Cross" in signals and base.get('lorentzian', 0) > 70:
            setup_found = True
            
    elif pipe_type == 4: # RSI Divergence + Candlestick Reversal
        pipe_name = "Mean Reversion Pivot"
        if any("Divergence" in s for s in signals) and any("Candlestick" in s for s in signals):
            setup_found = True

    if setup_found:
        base['signals'].append(f"PIPED: {pipe_name}")
        base['prediction_score'] += 10 # Bonus for meeting composite setup
        
    return base
