#!/usr/bin/env python3
"""
Full Market Backtest Orchestrator
==================================
Runs ALL strategies on the full universe, separately:

1. Extended Signals (RSI, MACD, Golden Cross, VSA, Lorentzian, PSAR, etc.)
2. Factor Rankings (Alpha, Momentum, Low Vol, Quality, Value, Multifactor)
3. VCP / Trend Template (Minervini Phase 2 + VCP pattern detection)
4. Advanced Portfolio Simulation (Phase 2 entry/exit with trailing stops)

Usage:
    python run_full_backtest.py                              # Default: NIFTY 500
    python run_full_backtest.py --universe "NIFTY 100"       # Specific index
    python run_full_backtest.py --universe FULL               # ALL NSE stocks
    python run_full_backtest.py --test                        # Quick test (20 stocks)
    python run_full_backtest.py --skip-factors                # Skip slow factor backtest
"""

import argparse
import logging
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_strategy(name, cmd, cwd=None):
    """Run a strategy backtest as a subprocess."""
    print(f"\n{'━' * 80}")
    print(f"  🚀 STRATEGY: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'━' * 80}\n")

    try:
        result = subprocess.run(
            cmd, cwd=cwd,
            timeout=7200  # 2 hour timeout per strategy
        )
        if result.returncode == 0:
            logger.info(f"  ✅ {name} completed successfully")
            return True
        else:
            logger.error(f"  ❌ {name} exited with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"  ❌ {name} timed out (2h limit)")
        return False
    except Exception as e:
        logger.error(f"  ❌ {name} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Full Market Backtest Orchestrator')
    parser.add_argument('--universe', type=str, default='NIFTY 500',
                        help='Universe: NIFTY 50, NIFTY 100, NIFTY 500, FULL')
    parser.add_argument('--days-ago', type=int, default=30,
                        help='Days ago for signal-based backtests')
    parser.add_argument('--forward', type=int, default=60,
                        help='Forward period for factor backtests')
    parser.add_argument('--test', action='store_true', help='Quick test mode')
    parser.add_argument('--skip-factors', action='store_true',
                        help='Skip factor ranking backtest (slow due to API calls)')
    parser.add_argument('--skip-portfolio', action='store_true',
                        help='Skip advanced portfolio simulation')
    args = parser.parse_args()

    start_time = datetime.now()

    print("=" * 80)
    print("  FULL MARKET BACKTEST ORCHESTRATOR")
    print(f"  Universe: {args.universe}")
    print(f"  Signal lookback: {args.days_ago} days ago")
    print(f"  Factor forward period: {args.forward} days")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    py = sys.executable
    universe_flag = ['--full'] if args.universe.upper() == 'FULL' else ['--index', args.universe]
    test_flag = ['--test'] if args.test else []

    results = {}

    # ─── 1. Extended Signals Backtest ────────────────────────────────────
    cmd = [py, 'backtest_extended_signals.py'] + universe_flag + test_flag + [
        '--days-ago', str(args.days_ago)
    ]
    results['Extended Signals'] = run_strategy('Extended Signals (RSI, MACD, VSA, Lorentzian, etc.)', cmd)

    # ─── 2. VCP / Trend Template Backtest ────────────────────────────────
    cmd = [py, 'backtest_vcp.py'] + universe_flag + test_flag + [
        '--days-ago', str(args.days_ago)
    ]
    results['VCP / Trend Template'] = run_strategy('VCP / Minervini Trend Template', cmd)

    # ─── 3. Factor Ranking Backtest ──────────────────────────────────────
    if not args.skip_factors:
        cmd = [py, 'backtest_factor_ranking.py'] + universe_flag + test_flag + [
            '--all-factors',
            '--forward', str(args.forward)
        ]
        results['Factor Rankings'] = run_strategy(
            'Factor Rankings (Alpha, Momentum, Quality, Value, Multifactor)', cmd
        )
    else:
        logger.info("Skipping factor ranking backtest (--skip-factors)")
        results['Factor Rankings'] = None

    # ─── 4. Advanced Portfolio Simulation ────────────────────────────────
    if not args.skip_portfolio:
        portfolio_universe = args.universe
        symbols_count = '50' if args.test else '500'
        cmd = [py, 'run_advanced_backtest.py',
               '--days', '365',
               '--symbols', symbols_count]
        if args.universe.upper() != 'FULL':
            cmd += ['--index', args.universe]
        results['Portfolio Simulation'] = run_strategy(
            'Advanced Portfolio Simulation (Phase 2 Entry/Exit)', cmd
        )
    else:
        logger.info("Skipping portfolio simulation (--skip-portfolio)")
        results['Portfolio Simulation'] = None

    # ─── Summary ─────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n\n{'=' * 80}")
    print("  FULL BACKTEST ORCHESTRATOR — FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Universe: {args.universe}")
    print(f"  Total Time: {elapsed/60:.1f} minutes")
    print(f"  Reports saved to: ./data/reports/")
    print()

    for name, success in results.items():
        if success is True:
            print(f"  ✅ {name}")
        elif success is False:
            print(f"  ❌ {name} (FAILED)")
        else:
            print(f"  ⏭️  {name} (SKIPPED)")

    print(f"\n{'=' * 80}")
    print("  🏁 ALL BACKTESTS COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
