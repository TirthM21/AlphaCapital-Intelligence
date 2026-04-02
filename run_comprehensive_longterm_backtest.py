#!/usr/bin/env python3
"""Run all major backtest families and write one consolidated markdown report."""

import argparse
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.data.universe_fetcher import StockUniverseFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TIME_HORIZONS = {
    "1 Year": 365,
    "3 Years": 1095,
    "5 Years": 1825,
    "10 Years": 3650,
    "15 Years": 5475,
    "20 Years": 7300,
}


def run_command(cmd: List[str]) -> str:
    """Run a subprocess and return its combined stdout/stderr."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    if result.returncode != 0:
        logger.error("Command failed with code %s", result.returncode)
        if output.strip():
            logger.error("Command output:\n%s", output)
    return output


def extract_metric(output: str, label: str) -> float:
    """Extract the first numeric metric from a labelled line."""
    for line in output.splitlines():
        if label in line:
            match = re.search(r"(-?\d+(?:\.\d+)?)", line.replace(",", ""))
            if match:
                return float(match.group(1))
    return 0.0


def parse_factor_summary(output: str) -> List[Dict[str, object]]:
    """Parse the comparison table emitted by backtest_factor_ranking.py."""
    rows: List[Dict[str, object]] = []
    pattern = re.compile(
        r"^\s*[^\s]+\s+([a-z_]+)\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)%\s+"
        r"([+-]?\d+\.\d+)%\s+(\d+\.\d+)%\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)%\s+"
        r"([+-]?\d+\.\d+)%\s+(\d+)\s*$"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rows.append(
            {
                "factor": match.group(1),
                "return": float(match.group(2)),
                "benchmark": float(match.group(3)),
                "alpha": float(match.group(4)),
                "win_rate": float(match.group(5)),
                "median": float(match.group(6)),
                "best": float(match.group(7)),
                "worst": float(match.group(8)),
                "stocks": int(match.group(9)),
            }
        )
    return rows


def parse_signal_summary(output: str, label: str) -> Dict[str, object]:
    """Parse a summary block from the signal-based backtests."""
    lines = output.splitlines()
    block_start = None
    for idx, line in enumerate(lines):
        if label in line:
            block_start = idx
            break
    if block_start is None:
        return {"label": label, "count": 0, "periods": {}}

    count_match = re.search(r"\((\d+)\s+signals\)", lines[block_start])
    summary: Dict[str, object] = {
        "label": label,
        "count": int(count_match.group(1)) if count_match else 0,
        "periods": {},
    }

    period_pattern = re.compile(
        r"^\s*(\d+D): Avg\s+([+-]?\d+\.\d+)%\s+Med\s+([+-]?\d+\.\d+)%\s+"
        r"WinRate\s+(\d+\.\d+)%\s+Best\s+([+-]?\d+\.\d+)%\s+Worst\s+([+-]?\d+\.\d+)%"
    )
    for line in lines[block_start + 1 :]:
        match = period_pattern.match(line)
        if match:
            summary["periods"][match.group(1)] = {
                "avg": float(match.group(2)),
                "median": float(match.group(3)),
                "win_rate": float(match.group(4)),
                "best": float(match.group(5)),
                "worst": float(match.group(6)),
            }
        elif summary["periods"] and line.strip().startswith("="):
            break
    return summary


def parse_vcp_summary(output: str) -> Dict[str, object]:
    """Parse the VCP summary block."""
    scanned_match = re.search(r"Scanned:\s+(\d+)\s+stocks\s+\|\s+Signals found:\s+(\d+)", output)
    summary: Dict[str, object] = {
        "scanned": int(scanned_match.group(1)) if scanned_match else 0,
        "signals": int(scanned_match.group(2)) if scanned_match else 0,
        "periods": {},
    }
    pattern = re.compile(
        r"^\s*(\d+D)\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)%\s+(\d+\.\d+)%\s+([+-]?\d+\.\d+)%\s+([+-]?\d+\.\d+)%"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            summary["periods"][match.group(1)] = {
                "avg": float(match.group(2)),
                "median": float(match.group(3)),
                "win_rate": float(match.group(4)),
                "best": float(match.group(5)),
                "worst": float(match.group(6)),
            }
    return summary


def build_strategy_cmd(
    script: str,
    universe: str,
    full_universe: bool,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build a subprocess command for one backtest family."""
    cmd = [sys.executable, script]
    supports_full = {
        "run_advanced_backtest.py",
        "backtest_factor_ranking.py",
        "backtest_vcp.py",
        "backtest_extended_signals.py",
    }
    if full_universe and script in supports_full:
        cmd.append("--full")
    else:
        cmd.extend(["--index", universe])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def render_period_table(summary: Dict[str, object], periods: List[str]) -> str:
    """Render a markdown table for parsed signal summaries."""
    rows = [
        "| Period | Avg Return | Median | Win Rate | Best | Worst |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    values = summary.get("periods", {})
    for period in periods:
        item = values.get(period)
        if item:
            rows.append(
                f"| {period} | {item['avg']}% | {item['median']}% | {item['win_rate']}% | {item['best']}% | {item['worst']}% |"
            )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive long-term backtest runner")
    parser.add_argument("--universe", type=str, default="NIFTY 500", help='Universe name, e.g. "NIFTY 500"')
    parser.add_argument("--full", action="store_true", help="Use the full NSE universe")
    parser.add_argument("--symbols", type=int, default=0, help="Portfolio backtest universe cap (0 = full selected universe)")
    parser.add_argument("--factor-top", type=int, default=30, help="Top N stocks per factor")
    parser.add_argument("--signal-days-ago", type=int, default=30, help="Lookback offset for signal-style backtests")
    parser.add_argument("--factor-forward", type=int, default=60, help="Forward window for factor backtests")
    args = parser.parse_args()

    report_path = Path("./data/reports/comprehensive_backtest.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    universe_label = "FULL NSE" if args.full else args.universe
    fetcher = StockUniverseFetcher()
    actual_symbols = fetcher.fetch_universe(index_name=None if args.full else args.universe)
    symbol_count = len(actual_symbols) if args.symbols == 0 else min(len(actual_symbols), args.symbols)

    portfolio_results = []
    print(f"\nStarting comprehensive backtest on {universe_label} ({symbol_count} symbols in scope)...")

    for label, days in TIME_HORIZONS.items():
        print(f"  Running portfolio simulation: {label}...")
        cmd = build_strategy_cmd(
            "run_advanced_backtest.py",
            args.universe,
            args.full,
            ["--days", str(days), "--symbols", str(args.symbols), "--short"],
        )
        output = run_command(cmd)
        total_return = extract_metric(output, "Total Return:")
        benchmark = extract_metric(output, "Benchmark (Nifty 50):")
        drawdown = extract_metric(output, "Max Drawdown:")
        win_rate = extract_metric(output, "Win Rate:")
        cagr = extract_metric(output, "CAGR:")
        avg_trade = extract_metric(output, "Avg Trade Gain:")

        portfolio_results.append(
            {
                "horizon": label,
                "return": total_return,
                "benchmark": benchmark,
                "alpha": round(total_return - benchmark, 2),
                "drawdown": drawdown,
                "win_rate": win_rate,
                "cagr": cagr,
                "avg_trade": avg_trade,
            }
        )

    print("  Running factor ranking backtest...")
    factor_output = run_command(
        build_strategy_cmd(
            "backtest_factor_ranking.py",
            args.universe,
            args.full,
            ["--all-factors", "--top", str(args.factor_top), "--forward", str(args.factor_forward), "--short"],
        )
    )
    factor_results = parse_factor_summary(factor_output)

    print("  Running VCP backtest...")
    vcp_output = run_command(
        build_strategy_cmd(
            "backtest_vcp.py",
            args.universe,
            args.full,
            ["--days-ago", str(args.signal_days_ago), "--short"],
        )
    )
    vcp_summary = parse_vcp_summary(vcp_output)

    print("  Running extended signals backtest...")
    extended_output = run_command(
        build_strategy_cmd(
            "backtest_extended_signals.py",
            args.universe,
            args.full,
            ["--days-ago", str(args.signal_days_ago), "--short"],
        )
    )
    extended_all = parse_signal_summary(extended_output, "ALL EXTENDED SIGNALS")
    extended_bullish = parse_signal_summary(extended_output, "BULLISH SIGNALS ONLY")
    extended_bearish = parse_signal_summary(extended_output, "BEARISH SIGNALS ONLY")

    best_factor = max(factor_results, key=lambda row: row["alpha"], default=None)
    best_portfolio = max(portfolio_results, key=lambda row: row["alpha"], default=None)

    md_lines = [
        "# Comprehensive Trading Strategy Backtest Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Universe: {universe_label} ({symbol_count} symbols)",
        "",
        "## Executive Summary",
    ]
    if best_portfolio:
        md_lines.append(
            f"- Best portfolio horizon: **{best_portfolio['horizon']}** with **{best_portfolio['alpha']}% alpha** "
            f"and **{best_portfolio['cagr']}% CAGR**."
        )
    if best_factor:
        md_lines.append(
            f"- Best factor snapshot: **{best_factor['factor']}** with **{best_factor['alpha']}% alpha** "
            f"across **{best_factor['stocks']}** selected stocks."
        )
    md_lines.append(
        f"- VCP signals found: **{vcp_summary['signals']}** from **{vcp_summary['scanned']}** scanned names."
    )
    md_lines.append(
        f"- Extended signals: **{extended_all['count']}** total, **{extended_bullish['count']}** bullish, "
        f"**{extended_bearish['count']}** bearish."
    )
    md_lines.extend(
        [
            "",
            "## Portfolio Strategy",
            "",
            "| Horizon | Return | Benchmark | Alpha | CAGR | Max Drawdown | Win Rate | Avg Trade |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in portfolio_results:
        md_lines.append(
            f"| {row['horizon']} | {row['return']}% | {row['benchmark']}% | {row['alpha']}% | "
            f"{row['cagr']}% | {row['drawdown']}% | {row['win_rate']}% | {row['avg_trade']}% |"
        )

    md_lines.extend(["", "## Factor Strategies", ""])
    if factor_results:
        md_lines.extend(
            [
                "| Factor | Return | Benchmark | Alpha | Win Rate | Median | Best | Worst | Stocks |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in factor_results:
            md_lines.append(
                f"| {row['factor']} | {row['return']}% | {row['benchmark']}% | {row['alpha']}% | "
                f"{row['win_rate']}% | {row['median']}% | {row['best']}% | {row['worst']}% | {row['stocks']} |"
            )
    else:
        md_lines.append("No factor results were parsed from the subprocess output.")

    md_lines.extend(
        [
            "",
            "## VCP / Trend Template",
            "",
            f"Signals found: **{vcp_summary['signals']}** from **{vcp_summary['scanned']}** scanned stocks.",
            "",
            render_period_table(vcp_summary, ["1D", "5D", "10D", "22D", "30D"]),
            "",
            "## Extended Signals",
            "",
            f"All signals: **{extended_all['count']}**",
            "",
            render_period_table(extended_all, ["1D", "5D", "10D", "22D", "30D"]),
            "",
            f"Bullish-only signals: **{extended_bullish['count']}**",
            "",
            render_period_table(extended_bullish, ["1D", "5D", "10D", "22D", "30D"]),
            "",
            f"Bearish-only signals: **{extended_bearish['count']}**",
            "",
            render_period_table(extended_bearish, ["1D", "5D", "10D", "22D", "30D"]),
            "",
            "## Notes",
            "- The portfolio backtester now ranks candidates by template quality, relative-strength slope, volume confirmation, and VCP bonus before allocating capital.",
            "- Passing `--symbols 0` uses the full selected universe rather than truncating to a demo-sized subset.",
            "- Signal and factor sections are parsed into this markdown report instead of being run as opaque side jobs.",
            "",
            "---",
            "*Reports saved in `./data/reports/`*",
        ]
    )

    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nAll backtests complete. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
