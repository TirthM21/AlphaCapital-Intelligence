"""Performance backtesting engine for trading signals.

Calculates returns over multiple timeframes and generates performance tables.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def calculate_returns(df: pd.DataFrame, entry_index: int, periods: List[int]) -> Dict[str, float]:
    """Calculate returns over specific periods after an entry point.
    
    Args:
        df: Historical OHLCV data
        entry_index: Index of the signal/entry
        periods: List of days to look ahead
        
    Returns:
        Dict of results (e.g. {'1D': 0.02, '5D': 0.05})
    """
    if entry_index >= len(df) - 1:
        return {}
        
    entry_price = df['Close'].iloc[entry_index]
    results = {}
    
    for p in periods:
        target_idx = entry_index + p
        if target_idx < len(df):
            exit_price = df['Close'].iloc[target_idx]
            ret = (exit_price - entry_price) / entry_price
            results[f'{p}D'] = round(ret * 100, 2)
        else:
            # Not enough data for this period yet
            results[f'{p}D'] = None
            
    return results

def generate_growth_table(ticker: str, df: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    """Calculate the growth of 10k over time."""
    if df.empty:
        return pd.DataFrame()
        
    # Calculate daily returns
    daily_ret = df['Close'].pct_change().fillna(0)
    
    # Cumulative growth
    growth = (1 + daily_ret).cumprod() * initial_capital
    
    result_df = pd.DataFrame({
        'Date': df.index,
        'Close': df['Close'],
        'Value': growth.round(2)
    })
    
    return result_df

def format_performance_summary(results: List[Dict]):
    """Format the backtest results into a table as seen in screenshots."""
    if not results:
        return "No results to summarize."
        
    # Flatten results
    df = pd.DataFrame(results)
    
    # Define periods
    periods = [1, 5, 10, 22, 30]
    
    summary = []
    summary.append(f"{'Ticker':<12} | {'Signal Date':<12} | " + " | ".join([f"{str(p)+'D %':<8}" for p in periods]))
    summary.append("-" * 80)
    
    for _, row in df.iterrows():
        line = f"{row.get('ticker', 'N/A'):<12} | {str(row.get('date', 'N/A'))[:10]:<12} | "
        perf_line = []
        for p in periods:
            val = row.get(f'{p}D')
            if val is not None:
                perf_line.append(f"{val:>7.2f}%")
            else:
                perf_line.append(f"{'N/A':>8}")
        line += " | ".join(perf_line)
        summary.append(line)
        
    return "\n".join(summary)
