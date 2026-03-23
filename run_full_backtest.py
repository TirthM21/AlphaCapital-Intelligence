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
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_strategy(name, cmd, short=False):
    """Run a strategy backtest as a subprocess."""
    print(f"\n{'━' * 80}")
    print(f"  🚀 STRATEGY: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'━' * 80}\n")

    try:
        if short:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, bufsize=1, universal_newlines=True
            )
            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                if not any(k in line for k in ["INFO", "Downloading", "Processing", "["]):
                    print(line, end="") # Only print interesting lines in short mode
            process.wait(timeout=7200)
            returncode = process.returncode
        else:
            result = subprocess.run(cmd, timeout=7200)
            returncode = result.returncode

        if returncode == 0:
            logger.info(f"  ✅ {name} completed successfully")
            return True
        else:
            logger.error(f"  ❌ {name} exited with code {returncode}")
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
    parser.add_argument('--short', action='store_true', help='Summarize output (<100 lines per strat)')
    parser.add_argument('--skip-factors', action='store_true',
                        help='Skip factor ranking backtest (slow due to API calls)')
    parser.add_argument('--skip-portfolio', action='store_true',
                        help='Skip advanced portfolio simulation')
    args = parser.parse_args()

    start_time = datetime.now()

    print("=" * 80)
    print("  FULL MARKET BACKTEST ORCHESTRATOR")
    print(f"  Universe: {args.universe}")
    print(f"  Short output: {args.short}")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    py = sys.executable
    universe_flag = ['--full'] if args.universe.upper() == 'FULL' else ['--index', args.universe]
    test_flag = ['--test'] if args.test else []
    short_flag = ['--short'] if args.short else []

    results = {} # type: Dict[str, Optional[bool]]
    results['Extended Signals'] = None
    results['VCP / Trend Template'] = None
    results['Factor Rankings'] = None
    results['Portfolio Simulation'] = None

    # ─── 1. Extended Signals Backtest ────────────────────────────────────
    cmd = [py, 'backtest_extended_signals.py'] + universe_flag + test_flag + short_flag + [
        '--days-ago', str(args.days_ago)
    ]
    results['Extended Signals'] = run_strategy('Extended Signals', cmd, short=args.short)

    # ─── 2. VCP / Trend Template Backtest ────────────────────────────────
    cmd = [py, 'backtest_vcp.py'] + universe_flag + test_flag + short_flag + [
        '--days-ago', str(args.days_ago)
    ]
    results['VCP / Trend Template'] = run_strategy('VCP / Trend Template', cmd, short=args.short)

    # ─── 3. Factor Ranking Backtest ──────────────────────────────────────
    if not args.skip_factors:
        cmd = [py, 'backtest_factor_ranking.py'] + universe_flag + test_flag + short_flag + [
            '--all-factors',
            '--forward', str(args.forward)
        ]
        results['Factor Rankings'] = run_strategy('Factor Rankings', cmd, short=args.short)
    else:
        results['Factor Rankings'] = None

    # ─── 4. Advanced Portfolio Simulation ────────────────────────────────
    if not args.skip_portfolio:
        symbols_count = '50' if args.test else '500'
        cmd = [py, 'run_advanced_backtest.py',
               '--days', '365', short_flag[0] if short_flag else '',
               '--symbols', symbols_count]
        if args.universe.upper() != 'FULL':
            cmd += ['--index', args.universe]
        results['Portfolio Simulation'] = run_strategy('Portfolio Simulation', cmd, short=args.short)
    else:
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
