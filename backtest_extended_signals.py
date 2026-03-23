#!/usr/bin/env python3
"""
Backtest Extended Signals Strategy
===================================
Tests the extended_signals module (RSI, MACD, Golden Cross, VCP patterns,
Lorentzian, VSA, PSAR, TTM Squeeze, Keltner, Candlestick, etc.)
by looking back N days, scoring signals, then measuring forward returns.

Usage:
    python backtest_extended_signals.py --index "NIFTY 500"
    python backtest_extended_signals.py --full --days-ago 60
    python backtest_extended_signals.py --index "NIFTY 50" --test
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

from src.data.universe_fetcher import StockUniverseFetcher
from src.screening.extended_signals import score_extended_signals
from src.analysis.backtest_engine import calculate_returns, format_performance_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backtest_extended(tickers, days_ago=30, forward_periods=None):
    """Run backtest: find signals from `days_ago`, compute forward returns."""
    if forward_periods is None:
        forward_periods = [1, 5, 10, 22, 30]

    results = []
    bullish_results = []
    bearish_results = []
    
    # Ensure all tickers have a suffix (.NS default)
    tickers = [f"{t}.NS" if "." not in t else t for t in tickers]
    
    from src.data.fetcher import YahooFinanceFetcher
    fetcher = YahooFinanceFetcher()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info(f"Processing {len(tickers)} stocks (using batch download and threads)...")
    
    # 1. Batch Download in chunks
    chunk_size = 300
    all_data = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        logger.info(f"Downloading chunk {i//chunk_size + 1}/{(len(tickers)-1)//chunk_size + 1}...")
        chunk_data = fetcher.batch_download(chunk, period="2y")
        all_data.update(chunk_data)
        
    # 2. Parallel Analysis
    def analyze_ticker(ticker, df):
        try:
            if df is None or df.empty or len(df) < 200:
                return None
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if len(df) < days_ago + 100:
                return None

            # Slice history up to `days_ago`
            idx_signal = len(df) - days_ago - 1
            hist_df = df.iloc[:idx_signal + 1]

            # Generate signals on historical data
            sig = score_extended_signals(ticker, hist_df)

            if sig.get('error') or not sig.get('signals'):
                return None

            # Calculate forward returns from the signal date
            perf = calculate_returns(df, idx_signal, forward_periods)

            # Classify direction
            bullish_keywords = [
                'Bullish', 'Golden', 'Oversold', 'Higher High',
                'Support Bounce', 'MFI Low', 'CCI Low',
                'Lorentzian Classifier (Bullish', 'PSAR Trend Reversal (Bullish',
                'Aroon Bullish', 'Squeeze Breakout', 'Upper Keltner Breakout',
                '52-Week High Breakout', 'Volume Breakout'
            ]
            bearish_keywords = [
                'Bearish', 'Death', 'Overbought', 'Lower High',
                'MFI High', 'CCI High',
                'Lorentzian Classifier (Bearish', 'PSAR Trend Reversal (Bearish',
                'Lower Keltner Rejection'
            ]

            is_bullish = any(
                any(kw in s for kw in bullish_keywords)
                for s in sig['signals']
            )
            is_bearish = any(
                any(kw in s for kw in bearish_keywords)
                for s in sig['signals']
            )

            entry = {
                'ticker': ticker,
                'date': hist_df.index[-1],
                'price': sig['price'],
                'rsi': sig['rsi'],
                'prediction_score': sig['prediction_score'],
                'lorentzian': sig['lorentzian'],
                'signal_count': len(sig['signals']),
                'signals': ', '.join(sig['signals'][:5]),
                'direction': 'bullish' if is_bullish and not is_bearish else (
                    'bearish' if is_bearish and not is_bullish else 'mixed'
                ),
                'is_bullish': is_bullish and not is_bearish,
                'is_bearish': is_bearish and not is_bullish
            }
            entry.update(perf)
            return entry
        except Exception as e:
            logger.error(f"Error backtesting {ticker}: {e}")
            return None

    max_workers = 15
    logger.info(f"Analyzing {len(all_data)} stocks with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(analyze_ticker, t, d): t 
            for t, d in all_data.items()
        }
        
        for future in as_completed(future_to_ticker):
            res = future.result()
            if res:
                results.append(res)
                if res['is_bullish']:
                    bullish_results.append(res)
                elif res['is_bearish']:
                    bearish_results.append(res)

    return results, bullish_results, bearish_results


def generate_summary(results, label, periods):
    """Generate aggregate statistics for a list of results."""
    if not results:
        return f"\n{label}: No signals found.\n"

    df = pd.DataFrame(results)
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  {label} ({len(df)} signals)")
    lines.append(f"{'='*80}")

    for p in periods:
        col = f'{p}D'
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                win_rate = (valid > 0).mean() * 100
                avg_ret = valid.mean()
                med_ret = valid.median()
                best = valid.max()
                worst = valid.min()
                lines.append(
                    f"  {col:>4}: Avg {avg_ret:>+7.2f}%  Med {med_ret:>+7.2f}%  "
                    f"WinRate {win_rate:>5.1f}%  Best {best:>+7.2f}%  Worst {worst:>+7.2f}%"
                )

    # Signal frequency
    all_sigs = []
    for r in results:
        for s in r.get('signals', '').split(', '):
            if s:
                all_sigs.append(s)

    if all_sigs:
        from collections import Counter
        top_sigs = Counter(all_sigs).most_common(10)
        lines.append("\n  Top Signals:")
        for sig_name, count in top_sigs:
            lines.append(f"    {count:>3}x  {sig_name}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Backtest Extended Signals Strategy')
    parser.add_argument('--index', type=str, default='NIFTY 500', help='Index universe')
    parser.add_argument('--full', action='store_true', help='ALL NSE stocks')
    parser.add_argument('--days-ago', type=int, default=30, help='Days ago to check for signals')
    parser.add_argument('--test', action='store_true', help='Test mode (20 stocks)')
    parser.add_argument('--short', action='store_true', help='Summary only, skip detailed table')
    args = parser.parse_args()

    uf = StockUniverseFetcher()
    if args.full:
        tickers = uf.fetch_universe()
    else:
        tickers = uf.fetch_universe(index_name=args.index)

    if args.test:
        tickers = tickers[:20]

    logger.info(f"Starting Extended Signals Backtest on {len(tickers)} stocks, {args.days_ago} days ago...")

    forward_periods = [1, 5, 10, 22, 30]
    all_results, bullish, bearish = backtest_extended(tickers, args.days_ago, forward_periods)

    # Print summaries
    print(generate_summary(all_results, "ALL EXTENDED SIGNALS", forward_periods))
    print(generate_summary(bullish, "BULLISH SIGNALS ONLY", forward_periods))
    print(generate_summary(bearish, "BEARISH SIGNALS ONLY (short/avoid)", forward_periods))

    # Detailed table
    if all_results and not args.short:
        print("\n" + "=" * 80)
        print("DETAILED SIGNAL PERFORMANCE (from {} days ago)".format(args.days_ago))
        print("=" * 80)
        print(format_performance_summary(all_results))

    # Save report
    report_dir = Path("./data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f"backtest_extended_{ts}.csv"
    if all_results:
        pd.DataFrame(all_results).to_csv(report_path, index=False)
        logger.info(f"Results saved to {report_path}")

    print(f"\n✅ Extended Signals Backtest Complete — {len(all_results)} signals analyzed")


if __name__ == '__main__':
    main()
