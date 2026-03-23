#!/usr/bin/env python3
"""
Backtest Factor Ranking Strategies
====================================
Tests NSE factor index strategies (Alpha, Momentum, Low Volatility, Quality,
Value, Dual Momentum, Multifactor MQVLv) by selecting top-N stocks on a
historical date and measuring forward performance.

Usage:
    python backtest_factor_ranking.py --factor momentum --index "NIFTY 500"
    python backtest_factor_ranking.py --factor alpha --index "NIFTY 200" --top 30
    python backtest_factor_ranking.py --factor multifactor --index "NIFTY 100"
    python backtest_factor_ranking.py --all-factors --index "NIFTY 500"
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import yfinance as yf

from src.data.universe_fetcher import StockUniverseFetcher
from src.analysis.factor_indices import NiftyFactorAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ALL_FACTORS = ['alpha', 'momentum', 'volatility', 'beta',
               'quality', 'value', 'dual_momentum', 'multifactor']

FACTOR_SORT = {
    'alpha':           ('alpha_score',          False),
    'momentum':        ('momentum_score',       False),
    'volatility':      ('low_vol_score',        False),  # Higher score = lower vol
    'beta':            ('beta',                 False),
    'quality':         ('quality_score',        False),
    'value':           ('value_score',          False),
    'dual_momentum':   ('dual_momentum_score',  False),
    'multifactor':     ('composite_score',      False),
}


def fetch_all_data(symbols, start_dt, end_dt, batch_size=50):
    """Download price history for all symbols in batches."""
    all_data = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tickers = [s if s.endswith('.NS') else f"{s}.NS" for s in batch]
        try:
            raw = yf.download(tickers, start=start_dt, end=end_dt,
                              group_by='ticker', threads=True, progress=False)
            for s in batch:
                t = s if s.endswith('.NS') else f"{s}.NS"
                try:
                    if len(tickers) == 1:
                        sdf = raw.copy()
                    else:
                        sdf = raw[t] if t in raw.columns.get_level_values(0) else pd.DataFrame()
                    if isinstance(sdf.columns, pd.MultiIndex):
                        sdf.columns = sdf.columns.get_level_values(0)
                    sdf = sdf.dropna()
                    if not sdf.empty and len(sdf) >= 100:
                        all_data[s] = sdf
                except Exception:
                    pass
            logger.info(f"  Batch {i // batch_size + 1} — {len(all_data)} stocks loaded")
        except Exception as e:
            logger.error(f"  Batch error: {e}")
    return all_data


def run_factor_backtest(factor, symbols, bench_df, all_data, total_days, top_n,
                        selection_offset_days=60):
    """
    Backtest a single factor:
    1. Use data up to (today - selection_offset_days) for factor scoring
    2. Select top N stocks
    3. Measure equal-weight portfolio return over the next selection_offset_days
    """
    # Split data: history for scoring vs forward for returns
    if bench_df.empty:
        return None

    dates = bench_df.index
    split_idx = len(dates) - selection_offset_days
    if split_idx < 252:
        logger.warning(f"Not enough history for {factor} backtest")
        return None

    scoring_end = dates[split_idx]
    forward_start = dates[split_idx + 1] if split_idx + 1 < len(dates) else dates[-1]

    # Build scoring data (up to scoring_end)
    scoring_data = {}
    for sym, df in all_data.items():
        hist = df.loc[:scoring_end]
        if len(hist) >= 252:
            scoring_data[sym] = hist

    if len(scoring_data) < 10:
        logger.warning(f"Only {len(scoring_data)} stocks have enough history for {factor}")
        return None

    scoring_bench = bench_df.loc[:scoring_end]

    # Compute factor scores
    analyzer = NiftyFactorAnalyzer()
    try:
        factor_df = analyzer.compute_scores(scoring_data, scoring_bench, factors=[factor])
    except Exception as e:
        logger.error(f"Factor computation failed for {factor}: {e}")
        return None

    if factor_df.empty:
        return None

    # Sort and select top N
    sort_col, ascending = FACTOR_SORT.get(factor, ('alpha', False))
    if sort_col not in factor_df.columns:
        logger.warning(f"Sort column '{sort_col}' not found for {factor}")
        return None

    ranked = factor_df.sort_values(by=sort_col, ascending=ascending)
    selected = ranked.head(top_n)
    selected_symbols = selected['symbol'].tolist()

    logger.info(f"  {factor.upper()}: Selected {len(selected_symbols)} stocks (top by {sort_col})")

    # Calculate forward returns for the selected portfolio
    forward_returns = []
    for sym in selected_symbols:
        if sym in all_data:
            fwd = all_data[sym].loc[forward_start:]
            if len(fwd) >= 2:
                entry_price = fwd['Close'].iloc[0]
                # Multi-period returns
                periods = {'5D': 5, '10D': 10, '22D': 22, '44D': 44}
                sym_result = {'symbol': sym, 'entry_price': entry_price}
                for label, days in periods.items():
                    if days < len(fwd):
                        exit_price = fwd['Close'].iloc[days]
                        sym_result[label] = float((exit_price - entry_price) / entry_price * 100)
                    else:
                        # Use last available
                        exit_price = fwd['Close'].iloc[-1]
                        sym_result[label] = float((exit_price - entry_price) / entry_price * 100)

                # Total return over available forward period
                final_price = fwd['Close'].iloc[-1]
                sym_result['total_return'] = float((final_price - entry_price) / entry_price * 100)
                forward_returns.append(sym_result)

    if not forward_returns:
        return None

    ret_df = pd.DataFrame(forward_returns)

    # Benchmark forward return
    bench_fwd = bench_df.loc[forward_start:]
    if len(bench_fwd) >= 2:
        bench_return = float((bench_fwd['Close'].iloc[-1] - bench_fwd['Close'].iloc[0]) /
                             bench_fwd['Close'].iloc[0] * 100)
    else:
        bench_return = 0.0

    # Portfolio stats (equal-weight)
    portfolio_return = ret_df['total_return'].mean()
    alpha = portfolio_return - bench_return

    result = {
        'factor': factor,
        'top_n': top_n,
        'num_stocks': len(ret_df),
        'selection_date': str(scoring_end.date()),
        'forward_days': selection_offset_days,
        'portfolio_return': round(portfolio_return, 2),
        'benchmark_return': round(bench_return, 2),
        'alpha': round(alpha, 2),
        'win_rate': round((ret_df['total_return'] > 0).mean() * 100, 1),
        'best': round(ret_df['total_return'].max(), 2),
        'worst': round(ret_df['total_return'].min(), 2),
        'median': round(ret_df['total_return'].median(), 2),
        'selected_stocks': selected_symbols[:10],  # Show top 10
        'details': ret_df
    }

    # Period-wise stats
    for p in ['5D', '10D', '22D', '44D']:
        if p in ret_df.columns:
            valid = ret_df[p].dropna()
            if len(valid) > 0:
                result[f'{p}_avg'] = round(valid.mean(), 2)
                result[f'{p}_win_rate'] = round((valid > 0).mean() * 100, 1)

    return result


def format_factor_report(results: List[Dict]) -> str:
    """Format factor backtest results into a comparison table."""
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("  FACTOR STRATEGY BACKTEST COMPARISON")
    lines.append("=" * 100)

    header = (f"  {'Factor':<18} {'Return':>8} {'Bench':>8} {'Alpha':>8} "
              f"{'WinRate':>8} {'Median':>8} {'Best':>8} {'Worst':>8} {'Stocks':>7}")
    lines.append(header)
    lines.append("  " + "-" * 94)

    for r in sorted(results, key=lambda x: x['alpha'], reverse=True):
        alpha_emoji = "🟢" if r['alpha'] > 0 else "🔴"
        line = (
            f"  {alpha_emoji} {r['factor']:<15} {r['portfolio_return']:>+7.2f}% "
            f"{r['benchmark_return']:>+7.2f}% {r['alpha']:>+7.2f}% "
            f"{r['win_rate']:>6.1f}% {r['median']:>+7.2f}% "
            f"{r['best']:>+7.2f}% {r['worst']:>+7.2f}% {r['num_stocks']:>5}"
        )
        lines.append(line)

    lines.append("\n  Period-by-Period Average Returns:")
    lines.append("  " + "-" * 70)

    period_header = f"  {'Factor':<18}"
    for p in ['5D', '10D', '22D', '44D']:
        period_header += f" {p+' Avg':>10} {p+' WR':>8}"
    lines.append(period_header)
    lines.append("  " + "-" * 70)

    for r in results:
        line = f"  {r['factor']:<18}"
        for p in ['5D', '10D', '22D', '44D']:
            avg = r.get(f'{p}_avg', 'N/A')
            wr = r.get(f'{p}_win_rate', 'N/A')
            if isinstance(avg, (int, float)):
                line += f" {avg:>+9.2f}% {wr:>6.1f}%"
            else:
                line += f" {'N/A':>10} {'N/A':>8}"
        lines.append(line)

    lines.append("\n" + "=" * 100)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Backtest Factor Ranking Strategies')
    parser.add_argument('--index', type=str, default='NIFTY 500', help='Index universe')
    parser.add_argument('--factor', type=str, default=None, choices=ALL_FACTORS,
                        help='Single factor to test')
    parser.add_argument('--all-factors', action='store_true', help='Test ALL factors')
    parser.add_argument('--top', type=int, default=30, help='Top N stocks to select')
    parser.add_argument('--days', type=int, default=500, help='Total historical data window')
    parser.add_argument('--forward', type=int, default=60,
                        help='Forward period (days) for return measurement')
    parser.add_argument('--test', action='store_true', help='Test mode (50 stocks)')
    parser.add_argument('--short', action='store_true', help='Summarize output (<100 lines)')
    args = parser.parse_args()

    factors_to_test = ALL_FACTORS if args.all_factors else ([args.factor] if args.factor else ['momentum'])

    logger.info(f"Factor Backtest: {', '.join(factors_to_test)} | Universe: {args.index} | Top {args.top}")

    # 1. Fetch universe
    uf = StockUniverseFetcher()
    symbols = uf.fetch_universe(index_name=args.index)
    if args.test:
        symbols = symbols[:50]

    if not symbols:
        logger.error("No symbols found"); return

    # 2. Download benchmark
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=args.days + 30)
    logger.info("Downloading benchmark (Nifty 50)...")
    bench = yf.download('^NSEI', start=start_dt, end=end_dt, progress=False)
    if bench.empty:
        logger.error("Benchmark download failed!"); return
    if isinstance(bench.columns, pd.MultiIndex):
        bench.columns = bench.columns.get_level_values(0)

    # 3. Download all stock data
    logger.info(f"Downloading {len(symbols)} stocks...")
    all_data = fetch_all_data(symbols, start_dt, end_dt)
    logger.info(f"Total stocks with data: {len(all_data)}")

    # 4. Run backtests for each factor
    factor_results = []
    for factor in factors_to_test:
        logger.info(f"\n{'━'*60}")
        logger.info(f"  Testing: {factor.upper()}")
        logger.info(f"{'━'*60}")

        result = run_factor_backtest(
            factor, symbols, bench, all_data,
            total_days=args.days, top_n=args.top,
            selection_offset_days=args.forward
        )

        if result:
            factor_results.append(result)
            logger.info(
                f"  ✅ {factor}: Return {result['portfolio_return']:+.2f}% "
                f"(Bench {result['benchmark_return']:+.2f}%, "
                f"Alpha {result['alpha']:+.2f}%)"
            )
        else:
            logger.warning(f"  ❌ {factor}: No results")

    # 5. Print comparison
    if factor_results:
        report = format_factor_report(factor_results)
        print(report)

        # Save
        report_dir = Path("./data/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = report_dir / f"backtest_factors_{ts}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"\nReport saved to {report_path}")

        # Save detailed CSV
        for r in factor_results:
            csv_path = report_dir / f"backtest_{r['factor']}_{ts}.csv"
            r['details'].to_csv(csv_path, index=False)

    print(f"\n✅ Factor Backtest Complete — {len(factor_results)} factors tested")


if __name__ == '__main__':
    main()
