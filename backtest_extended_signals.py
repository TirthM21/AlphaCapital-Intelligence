#!/usr/bin/env python3
"""
Backtest Extended Signals Strategy
===================================
Tests the extended_signals module and the composite piped setups by looking
back N days, generating setups on that historical date, and measuring forward
returns by strategy bucket.
"""

import argparse
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.analysis.backtest_engine import calculate_returns, format_performance_summary
from src.data.universe_fetcher import StockUniverseFetcher
from src.screening.extended_signals import score_extended_signals
from src.screening.piped_scanners import run_piped_scan

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PIPE_LABELS = {
    1: "PIPE 1: High Volume Momentum Breakout",
    2: "PIPE 2: VCP & Chart Pattern Setup",
    3: "PIPE 3: Institutional Trend Setup",
    4: "PIPE 4: Mean Reversion Pivot",
}


def classify_strategy_buckets(signal_result: Dict) -> List[str]:
    """Map technical signals into strategy buckets for performance testing."""
    signals = signal_result.get("signals", [])
    bullish_score = signal_result.get("bullish_score", 0)
    bearish_score = signal_result.get("bearish_score", 0)
    buckets = []

    breakout_signals = (
        "52-Week High Breakout",
        "Upper Keltner Breakout",
        "Volume Breakout",
        "Squeeze Breakout",
        "TTM Squeeze",
    )
    trend_signals = (
        "Golden Cross",
        "Bullish 9/21 EMA Cross",
        "Bullish MACD Cross",
        "MACD Momentum Improving",
        "Higher High / Higher Low Trend",
    )
    mean_reversion_signals = (
        "Bullish RSI Divergence",
        "RSI Oversold",
        "Potential Support Bounce Zone",
        "Candlestick: Hammer",
        "Candlestick: Morning Star",
        "Candlestick: Bullish Engulfing",
    )
    avoid_signals = (
        "Death Cross",
        "Bearish MACD Cross",
        "MACD Momentum Deteriorating",
        "RSI Overbought",
        "52-Week Low",
        "Lower High / Lower Low Trend",
    )

    has_breakout = any(any(token in signal for token in breakout_signals) for signal in signals)
    has_trend = any(any(token in signal for token in trend_signals) for signal in signals)
    has_mean_reversion = any(any(token in signal for token in mean_reversion_signals) for signal in signals)
    has_avoid = any(any(token in signal for token in avoid_signals) for signal in signals)
    has_lower_trend = any("Lower High / Lower Low Trend" in signal for signal in signals)
    has_support = any(token in signal for signal in signals for token in ("Potential Support Bounce Zone", "Near Fibonacci Level", "MFI Low", "CCI Low"))
    has_bullish_candle = any(
        token in signal
        for signal in signals
        for token in ("Candlestick: Hammer", "Candlestick: Morning Star", "Candlestick: Bullish Engulfing")
    )

    if bullish_score >= bearish_score + 2 and has_breakout and has_trend and not has_lower_trend and "52-Week Low" not in " | ".join(signals):
        buckets.append("Bullish Breakout")
    if bullish_score > bearish_score and has_trend and not has_lower_trend:
        buckets.append("Trend Following")
    if bullish_score >= bearish_score + 1 and has_mean_reversion and has_support and has_bullish_candle and not has_lower_trend:
        buckets.append("Mean Reversion")
    if bearish_score > bullish_score and has_avoid:
        buckets.append("Bearish / Avoid")

    return buckets


def build_entry(ticker: str, hist_df: pd.DataFrame, sig: Dict, perf: Dict[str, float], signal_preview_count: int = 5) -> Dict:
    """Normalize a signal result into a backtest row."""
    bullish_score = sig.get('bullish_score', 0)
    bearish_score = sig.get('bearish_score', 0)
    is_bullish = bullish_score > bearish_score
    is_bearish = bearish_score > bullish_score

    entry = {
        'ticker': ticker,
        'date': hist_df.index[-1],
        'price': sig['price'],
        'rsi': sig['rsi'],
        'prediction_score': sig['prediction_score'],
        'lorentzian': sig['lorentzian'],
        'bullish_score': bullish_score,
        'bearish_score': bearish_score,
        'signal_count': len(sig['signals']),
        'signals': ', '.join(sig['signals'][:signal_preview_count]),
        'direction': 'bullish' if is_bullish and not is_bearish else (
            'bearish' if is_bearish and not is_bullish else 'mixed'
        ),
        'is_bullish': is_bullish and not is_bearish,
        'is_bearish': is_bearish and not is_bullish,
    }
    entry.update(perf)
    return entry


def backtest_extended(tickers, days_ago=30, forward_periods=None, cached_only: bool = False):
    """Run performance-focused backtest across signal families and piped setups."""
    if forward_periods is None:
        forward_periods = [1, 5, 10, 22, 30]

    results = []
    bullish_results = []
    bearish_results = []
    strategy_results = {
        "Bullish Breakout": [],
        "Trend Following": [],
        "Mean Reversion": [],
        "Bearish / Avoid": [],
    }
    pipe_results = {label: [] for label in PIPE_LABELS.values()}

    tickers = [f"{ticker}.NS" if "." not in ticker else ticker for ticker in tickers]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.data.fetcher import YahooFinanceFetcher

    fetcher = YahooFinanceFetcher()
    logger.info("Processing %s stocks (using batch download and threads)...", len(tickers))

    chunk_size = 300
    all_data = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        logger.info("Downloading chunk %s/%s...", i // chunk_size + 1, (len(tickers) - 1) // chunk_size + 1)
        all_data.update(fetcher.batch_download(chunk, period="2y", cached_only=cached_only))

    def analyze_ticker(ticker, df):
        try:
            if df is None or df.empty or len(df) < 200:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < days_ago + 100:
                return None

            idx_signal = len(df) - days_ago - 1
            hist_df = df.iloc[:idx_signal + 1]
            sig = score_extended_signals(ticker, hist_df)
            if sig.get('error') or not sig.get('signals'):
                return None

            perf = calculate_returns(df, idx_signal, forward_periods)
            base_entry = build_entry(ticker, hist_df, sig, perf)

            bucket_entries = []
            for bucket in classify_strategy_buckets(sig):
                bucket_entry = dict(base_entry)
                bucket_entry['strategy'] = bucket
                bucket_entries.append((bucket, bucket_entry))

            pipe_entries = []
            for pipe_id, pipe_label in PIPE_LABELS.items():
                pipe_sig = run_piped_scan(ticker, hist_df, pipe_id)
                if pipe_sig.get('setup_found'):
                    pipe_entry = build_entry(ticker, hist_df, pipe_sig, perf, signal_preview_count=6)
                    pipe_entry['strategy'] = pipe_label
                    pipe_entries.append((pipe_label, pipe_entry))

            return {
                'entry': base_entry,
                'bucket_entries': bucket_entries,
                'pipe_entries': pipe_entries,
            }
        except Exception as exc:
            logger.error("Error backtesting %s: %s", ticker, exc)
            return None

    max_workers = 15
    logger.info("Analyzing %s stocks with %s workers...", len(all_data), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, ticker, df): ticker for ticker, df in all_data.items()}
        for future in as_completed(future_to_ticker):
            res = future.result()
            if not res:
                continue
            entry = res['entry']
            results.append(entry)
            if entry['is_bullish']:
                bullish_results.append(entry)
            elif entry['is_bearish']:
                bearish_results.append(entry)
            for bucket, bucket_entry in res['bucket_entries']:
                strategy_results[bucket].append(bucket_entry)
            for pipe_label, pipe_entry in res['pipe_entries']:
                pipe_results[pipe_label].append(pipe_entry)

    return results, bullish_results, bearish_results, strategy_results, pipe_results


def generate_summary(results, label, periods):
    """Generate aggregate statistics for a list of results."""
    if not results:
        return f"\n{label}: No setups found.\n"

    df = pd.DataFrame(results)
    lines = [f"\n{'=' * 80}", f"  {label} ({len(df)} setups)", f"{'=' * 80}"]

    for period in periods:
        col = f'{period}D'
        valid = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        if len(valid) == 0:
            continue
        win_rate = (valid > 0).mean() * 100
        avg_ret = valid.mean()
        med_ret = valid.median()
        best = valid.max()
        worst = valid.min()
        lines.append(
            f"  {col:>4}: Avg {avg_ret:>+7.2f}%  Med {med_ret:>+7.2f}%  "
            f"WinRate {win_rate:>5.1f}%  Best {best:>+7.2f}%  Worst {worst:>+7.2f}%"
        )

    all_sigs = []
    for row in results:
        all_sigs.extend([signal for signal in row.get('signals', '').split(', ') if signal])
    if all_sigs:
        lines.append("\n  Top Signals:")
        for sig_name, count in Counter(all_sigs).most_common(8):
            lines.append(f"    {count:>3}x  {sig_name}")

    return '\n'.join(lines)


def generate_strategy_summary(strategy_results: Dict[str, List[Dict]], periods: List[int]) -> str:
    """Generate per-strategy performance summaries."""
    blocks = []
    for label, entries in strategy_results.items():
        if entries:
            blocks.append(generate_summary(entries, label, periods))
    return "\n".join(blocks) if blocks else "\nNo qualifying strategies found.\n"


def main():
    parser = argparse.ArgumentParser(description='Backtest Extended Signals Strategy')
    parser.add_argument('--index', type=str, default='NIFTY 500', help='Index universe')
    parser.add_argument('--full', action='store_true', help='ALL NSE stocks')
    parser.add_argument('--days-ago', type=int, default=30, help='Days ago to check for signals')
    parser.add_argument('--test', action='store_true', help='Test mode (20 stocks)')
    parser.add_argument('--short', action='store_true', help='Summary only, skip detailed table')
    parser.add_argument('--cached-only', action='store_true', help='Use cached universe and price data only')
    args = parser.parse_args()

    uf = StockUniverseFetcher()
    tickers = (
        uf.fetch_universe(cached_only=args.cached_only)
        if args.full
        else uf.fetch_universe(index_name=args.index, cached_only=args.cached_only)
    )
    if args.test:
        tickers = tickers[:20]

    logger.info(
        "Starting Extended Signals Strategy Backtest on %s stocks, %s days ago...",
        len(tickers),
        args.days_ago,
    )

    forward_periods = [1, 5, 10, 22, 30]
    all_results, bullish, bearish, strategy_results, pipe_results = backtest_extended(
        tickers,
        args.days_ago,
        forward_periods,
        args.cached_only,
    )

    print(generate_summary(all_results, "ALL EXTENDED SIGNAL SNAPSHOTS", forward_periods))
    print(generate_summary(bullish, "BULLISH SETUPS ONLY", forward_periods))
    print(generate_summary(bearish, "BEARISH / AVOID SETUPS ONLY", forward_periods))
    print(generate_strategy_summary(strategy_results, forward_periods))
    print(generate_strategy_summary(pipe_results, forward_periods))

    if all_results and not args.short:
        print("\n" + "=" * 80)
        print(f"DETAILED SIGNAL PERFORMANCE (from {args.days_ago} days ago)")
        print("=" * 80)
        print(format_performance_summary(all_results))

    report_dir = Path("./data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f"backtest_extended_{ts}.csv"
    if all_results:
        export_rows = list(all_results)
        for entries in strategy_results.values():
            export_rows.extend(entries)
        for entries in pipe_results.values():
            export_rows.extend(entries)
        pd.DataFrame(export_rows).to_csv(report_path, index=False)
        logger.info("Results saved to %s", report_path)

    print(f"\nExtended Signals Strategy Backtest Complete: {len(all_results)} snapshots analyzed")


if __name__ == '__main__':
    main()
