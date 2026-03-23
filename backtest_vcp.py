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
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

from src.data.universe_fetcher import StockUniverseFetcher
from src.screening.phase_indicators import classify_phase, validate_minervini_trend_template
from src.screening.signal_engine import score_buy_signal
from src.analysis.backtest_engine import calculate_returns, format_performance_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backtest_vcp(tickers, days_ago=30, min_score=60, forward_periods=None):
    """
    For each ticker:
    1. Look at data from `days_ago` to apply Minervini Trend Template
    2. Score buy signal (VCP + Phase 2)
    3. Measure forward returns from the signal date
    """
    if forward_periods is None:
        forward_periods = [1, 5, 10, 22, 30]

    results = []
    total_scanned = 0

    for i, original_ticker in enumerate(tickers):
        try:
            ticker = original_ticker.strip().upper()
            if "." not in ticker:
                ticker += ".NS"

            logger.info(f"[{i+1}/{len(tickers)}] {ticker}")
            total_scanned += 1

            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty or len(df) < 200:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) < days_ago + 100:
                continue

            idx_signal = len(df) - days_ago - 1
            hist_df = df.iloc[:idx_signal + 1]

            # 1. Classify phase
            current_price = float(hist_df['Close'].iloc[-1])
            phase_info = classify_phase(hist_df, current_price)

            # 2. Check Minervini Trend Template
            sma_200_series = hist_df['Close'].rolling(200).mean()
            trend_info = validate_minervini_trend_template(current_price, phase_info, sma_200_series)

            if not trend_info.get('passes_template'):
                continue

            # 3. Score buy signal (includes VCP detection)
            # We need a dummy RS series for the scorer
            bench_close = hist_df['Close']  # Self-relative as placeholder
            rs_series = (bench_close / bench_close.rolling(50).mean()).dropna()

            sig = score_buy_signal(
                ticker=ticker,
                price_data=hist_df,
                current_price=current_price,
                phase_info=phase_info,
                rs_series=rs_series,
                fundamentals=None,
                vcp_data=phase_info.get('vcp_data')
            )

            if sig.get('score', 0) < min_score:
                continue

            logger.info(f"  ✓ Signal: score={sig['score']}, phase={sig['phase']}")

            # 4. Calculate forward returns
            perf = calculate_returns(df, idx_signal, forward_periods)

            entry = {
                'ticker': ticker,
                'date': hist_df.index[-1],
                'price': current_price,
                'score': sig['score'],
                'phase': sig['phase'],
                'entry_quality': sig.get('entry_quality', 'N/A'),
                'rr_ratio': sig.get('risk_reward_ratio', 0),
                'stop_loss': sig.get('stop_loss', 0),
                'minervini_score': sig.get('minervini_template_score', 0),
                'reasons': ' | '.join(sig.get('reasons', [])[:3]),
            }
            entry.update(perf)
            results.append(entry)

            time.sleep(0.3)

        except Exception as e:
            logger.error(f"Error backtesting {original_ticker}: {e}")

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
    args = parser.parse_args()

    uf = StockUniverseFetcher()
    if args.full:
        tickers = uf.fetch_universe()
    else:
        tickers = uf.fetch_universe(index_name=args.index)

    if args.test:
        tickers = tickers[:20]

    logger.info(f"Starting VCP Backtest on {len(tickers)} stocks, {args.days_ago} days ago...")

    forward_periods = [1, 5, 10, 22, 30]
    results, total_scanned = backtest_vcp(
        tickers, args.days_ago, args.min_score, forward_periods
    )

    # Summary
    summary = generate_vcp_summary(results, total_scanned, args.days_ago, forward_periods)
    print(summary)

    # Detailed table
    if results:
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

    print(f"\n✅ VCP Backtest Complete — {len(results)} signals from {total_scanned} stocks")


if __name__ == '__main__':
    main()
