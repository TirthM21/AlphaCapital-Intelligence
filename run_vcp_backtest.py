#!/usr/bin/env python3
"""VCP (Volatility Contraction Pattern) Backtesting Script.

Scans the market for historical VCP signals and calculates their performance.
"""

import argparse
import logging
import pandas as pd
import yfinance as yf
import time
from datetime import datetime
from typing import List, Dict

from src.data.universe_fetcher import StockUniverseFetcher
from src.screening.signal_engine import score_buy_signal
from src.screening.phase_indicators import validate_minervini_trend_template
from src.analysis.backtest_engine import calculate_returns, format_performance_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='VCP Strategy Backtester')
    parser.add_argument('--index', type=str, default='NIFTY 50', help='Index to scan')
    parser.add_argument('--full', action='store_true', help='Scan ALL NSE stocks')
    parser.add_argument('--test', action='store_true', help='Test with 10 stocks')
    parser.add_argument('--days_ago', type=int, default=30, help='Days ago to check for signals')
    
    args = parser.parse_args()
    
    universe_fetcher = StockUniverseFetcher()
    if args.full:
        tickers = universe_fetcher.nse.get_all_equity_stocks()
    else:
        tickers = universe_fetcher.fetch_universe(index_name=args.index)
    
    if args.test:
        tickers = tickers[:10]
        
    logger.info(f"Starting VCP Backtest for {len(tickers)} stocks...")
    
    results = []
    
    for original_ticker in tickers:
        try:
            # Clean and handle suffixes
            ticker = original_ticker.strip().upper()
            if "." not in ticker: ticker += ".NS"
            
            logger.info(f"Processing {ticker}...")
            
            # Fetch 2 years to have enough history for VCP 
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty or len(df) < 200:
                continue
                
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # We want to check signals from 'days_ago'
            if len(df) < args.days_ago + 100:
                continue
                
            idx_signal = len(df) - args.days_ago - 1
            hist_df = df.iloc[:idx_signal+1]
            
            # 1. Use classify_phase to get accurate technical state for that date
            from src.screening.phase_indicators import classify_phase
            current_price = hist_df['Close'].iloc[-1]
            phase_info = classify_phase(hist_df, current_price)
            
            # 2. Check Minervini Trend Template (VCP prerequisite)
            # The template requires 200 SMA series for slope
            sma_200_series = hist_df['Close'].rolling(200).mean()
            trend_info = validate_minervini_trend_template(current_price, phase_info, sma_200_series)
            
            if trend_info.get('passes_template'):
                # 3. Score VCP Signal
                res = score_buy_signal(ticker, hist_df, current_price, phase_info, phase=phase_info['phase'])
                
                if res.get('score', 0) >= 60:
                    logger.info(f"Signal found for {ticker} on {hist_df.index[-1].date()}!")
                    
                    # 4. Calculate returns from that point forward
                    perf = calculate_returns(df, idx_signal, [1, 5, 10, 22, 30])
                    
                    # Update result dict
                    res['ticker'] = ticker
                    res['date'] = hist_df.index[-1]
                    res.update(perf)
                    results.append(res)
                    
            time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error backtesting {original_ticker}: {e}")
            
    print("\n" + "="*80)
    print(f"VCP STRATEGY BACKTEST RESULTS ({args.days_ago} DAYS AGO)")
    print("="*80)
    if results:
        print(format_performance_summary(results))
    else:
        print("No VCP signals found in the specified period.")

if __name__ == '__main__':
    main()
