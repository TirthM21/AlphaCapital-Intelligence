#!/usr/bin/env python3
"""
Backtest VCP (Volatility Contraction Pattern) Strategy — Standalone
=====================================================================
Scans for Minervini Trend Template + VCP signals on a historical date,
then measures forward performance at 1D, 5D, 10D, 22D, 30D.

Usage:
    python backtest_vcp.py --index "NIFTY 500"
    python backtest_vcp.py --full --days-ago 60
    python backtest_vcp.py --index "NIFTY 50" --test
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

from src.data.fetcher import YahooFinanceFetcher
from src.data.universe_fetcher import StockUniverseFetcher
from src.screening.phase_indicators import (
    calculate_relative_strength,
    classify_phase,
    detect_vcp_pattern,
    validate_minervini_trend_template,
)
from src.screening.signal_engine import score_buy_signal
from src.analysis.backtest_engine import calculate_returns, format_performance_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _normalize_downloaded_frames(all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    normalized = {}
    for ticker, df in all_data.items():
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        normalized[ticker] = df.dropna()
    return normalized


def backtest_vcp(tickers, days_ago=30, min_score=60, forward_periods=None, cached_only: bool = False) -> Tuple[List[Dict], int]:
    """
    For each ticker:
    1. Look at data from `days_ago` to apply Minervini Trend Template
    2. Score buy signal (VCP + Phase 2)
    3. Measure forward returns from the signal date
    """
    if forward_periods is None:
        forward_periods = [1, 5, 10, 22, 30]

    results: List[Dict] = []
    tickers = [t.strip().upper() if "." in t else f"{t.strip().upper()}.NS" for t in tickers]
    total_scanned = len(tickers)

    fetcher = YahooFinanceFetcher()
    chunk_size = 300
    all_data: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        logger.info("Downloading chunk %s/%s...", i // chunk_size + 1, (len(tickers) - 1) // chunk_size + 1)
        chunk_data = fetcher.batch_download(chunk, period="2y", cached_only=cached_only)
        all_data.update(_normalize_downloaded_frames(chunk_data))

    benchmark = fetcher.batch_download(["^NSEI"], period="2y", cached_only=cached_only).get("^NSEI", pd.DataFrame())
    if isinstance(benchmark.columns, pd.MultiIndex):
        benchmark.columns = benchmark.columns.get_level_values(0)
    benchmark_close = benchmark["Close"] if not benchmark.empty else pd.Series(dtype=float)

    def analyze_ticker(ticker: str, df: pd.DataFrame) -> Dict | None:
        try:
            if df.empty or len(df) < 200:
                return None

            if len(df) < days_ago + 100:
                return None

            idx_signal = len(df) - days_ago - 1
            hist_df = df.iloc[:idx_signal + 1]

            current_price = float(hist_df['Close'].iloc[-1])
            phase_info = classify_phase(hist_df, current_price)
            vcp_data = detect_vcp_pattern(hist_df, current_price, phase_info)

            sma_200_series = hist_df['Close'].rolling(200).mean()
            trend_info = validate_minervini_trend_template(current_price, phase_info, sma_200_series)
            if not trend_info.get('passes_template'):
                return None

            rs_series = calculate_relative_strength(hist_df['Close'], benchmark_close) if not benchmark_close.empty else pd.Series(dtype=float)

            sig = score_buy_signal(
                ticker=ticker,
                price_data=hist_df,
                current_price=current_price,
                phase_info=phase_info,
                rs_series=rs_series,
                fundamentals=None,
                vcp_data=vcp_data
            )

            if sig.get('score', 0) < min_score:
                return None

            perf = calculate_returns(df, idx_signal, forward_periods)
            entry: Dict[str, object] = {
                'ticker': ticker,
                'date': hist_df.index[-1],
                'price': current_price,
                'score': sig['score'],
                'phase': sig['phase'],
                'entry_quality': sig.get('entry_quality', 'N/A'),
                'rr_ratio': sig.get('risk_reward_ratio', 0),
                'stop_loss': sig.get('stop_loss', 0),
                'minervini_score': sig.get('minervini_template_score', 0),
                'vcp_quality': round(vcp_data.get('vcp_quality', 0), 1),
                'vcp_pattern': vcp_data.get('pattern_details', ''),
                'reasons': ' | '.join(sig.get('reasons', [])[:3]),
            }
            entry.update(perf)
            return entry

        except Exception as e:
            logger.error(f"Error backtesting {ticker}: {e}")
            return None

    max_workers = 16
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_ticker, ticker, df): ticker for ticker, df in all_data.items()}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    return results, total_scanned


def generate_vcp_summary(results, total_scanned, days_ago, periods):
    """Generate formatted VCP backtest summary."""
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append(f"  VCP / TREND TEMPLATE BACKTEST RESULTS ({days_ago} DAYS AGO)")
    lines.append("=" * 90)
    lines.append(f"  Scanned: {total_scanned} stocks | Signals found: {len(results)}")
    lines.append(f"  Hit Rate: {len(results)/total_scanned*100:.1f}% of universe passed filters" if total_scanned > 0 else "")

    if not results:
        lines.append("\n  No VCP signals found in the specified period.")
        return '\n'.join(lines)

    df = pd.DataFrame(results)

    # Overall stats
    lines.append(f"\n  {'Period':>6} {'Avg Return':>12} {'Median':>10} {'Win Rate':>10} {'Best':>10} {'Worst':>10}")
    lines.append("  " + "-" * 62)

    for p in periods:
        col = f'{p}D'
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                lines.append(
                    f"  {col:>6} {valid.mean():>+11.2f}% {valid.median():>+9.2f}% "
                    f"{(valid > 0).mean()*100:>8.1f}% {valid.max():>+9.2f}% {valid.min():>+9.2f}%"
                )

    # Score-based bucketing
    if len(df) >= 5:
        lines.append(f"\n  Performance by Signal Score:")
        lines.append("  " + "-" * 50)
        for lo, hi, label in [(80, 200, '80+ (Strong)'), (70, 80, '70-80 (Good)'),
                               (60, 70, '60-70 (Marginal)')]:
            bucket = df[(df['score'] >= lo) & (df['score'] < hi)]
            if len(bucket) > 0 and '22D' in bucket.columns:
                valid_22d = bucket['22D'].dropna()
                if len(valid_22d) > 0:
                    lines.append(
                        f"  Score {label:>16}: {len(bucket):>3} signals, "
                        f"22D Avg {valid_22d.mean():>+.2f}%, WR {(valid_22d > 0).mean()*100:.0f}%"
                    )

    # Entry quality breakdown
    if 'entry_quality' in df.columns and '22D' in df.columns:
        lines.append(f"\n  Performance by Entry Quality:")
        lines.append("  " + "-" * 50)
        for eq in ['Good', 'Extended', 'Poor']:
            subset = df[df['entry_quality'] == eq]
            if len(subset) > 0:
                valid = subset['22D'].dropna()
                if len(valid) > 0:
                    lines.append(
                        f"  {eq:>10}: {len(subset):>3} signals, "
                        f"22D Avg {valid.mean():>+.2f}%, WR {(valid > 0).mean()*100:.0f}%"
                    )

    lines.append("\n" + "=" * 90)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Backtest VCP / Trend Template Strategy')
    parser.add_argument('--index', type=str, default='NIFTY 500', help='Index universe')
    parser.add_argument('--full', action='store_true', help='ALL NSE stocks')
    parser.add_argument('--days-ago', type=int, default=30, help='Days ago for signal check')
    parser.add_argument('--min-score', type=int, default=60, help='Minimum buy score threshold')
    parser.add_argument('--test', action='store_true', help='Test mode (20 stocks)')
    parser.add_argument('--short', action='store_true', help='Summary only, skip detailed table')
    parser.add_argument('--cached-only', action='store_true', help='Use cached universe and price data only')
    args = parser.parse_args()

    uf = StockUniverseFetcher()
    if args.full:
        tickers = uf.fetch_universe(cached_only=args.cached_only)
    else:
        tickers = uf.fetch_universe(index_name=args.index, cached_only=args.cached_only)

    if args.test:
        tickers = tickers[:20]

    logger.info(f"Starting VCP Backtest on {len(tickers)} stocks, {args.days_ago} days ago...")

    forward_periods = [1, 5, 10, 22, 30]
    results, total_scanned = backtest_vcp(
        tickers, args.days_ago, args.min_score, forward_periods, args.cached_only
    )

    # Summary
    summary = generate_vcp_summary(results, total_scanned, args.days_ago, forward_periods)
    print(summary)

    # Detailed table
    if results and not args.short:
        print("\n" + "=" * 90)
        print("INDIVIDUAL SIGNAL RESULTS")
        print("=" * 90)
        print(format_performance_summary(results))

    # Save
    report_dir = Path("./data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if results:
        pd.DataFrame(results).to_csv(report_dir / f"backtest_vcp_{ts}.csv", index=False)
        with open(report_dir / f"backtest_vcp_{ts}.txt", 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Reports saved to {report_dir}")

    print(f"\nVCP Backtest Complete: {len(results)} signals from {total_scanned} stocks")


if __name__ == '__main__':
    main()
