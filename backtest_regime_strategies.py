#!/usr/bin/env python3
"""
Backtest Regime Strategy Allocations
===================================
Uses the cached regime model and cached full-universe price data to evaluate
the regime-driven strategy stack on a historical date, then measures forward returns.
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.analysis.backtest_engine import calculate_returns, format_performance_summary
from src.analysis.regime_model import MarketRegimeDetector
from src.data.fetcher import YahooFinanceFetcher
from src.data.universe_fetcher import StockUniverseFetcher
from src.screening.regime_strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    ShortDefensiveStrategy,
    StrategyMetaController,
    VolatilityExpansionStrategy,
)
from src.screening.selection_engine import SelectionEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("data/cache/regime_model.pkl")
BENCHMARK_TICKER = "^NSEI"
SHORT_STRATEGIES = {"Short / Defensive"}


def load_regime_model() -> MarketRegimeDetector:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing cached regime model: {MODEL_PATH}")
    return MarketRegimeDetector.load_state(str(MODEL_PATH))


def build_signal_entry(ticker: str, hist_df: pd.DataFrame, alpha: Dict, perf: Dict[str, float], regime: str) -> Dict:
    best = alpha["best_signal"]
    entry = {
        "ticker": ticker,
        "date": hist_df.index[-1],
        "regime": regime,
        "strategy": alpha["primary_strategy"],
        "alpha_score": alpha["alpha_score"],
        "price": best["entry"],
        "stop_loss": best["stop_loss"],
        "target": best["target"],
        "expected_return": best["expected_return"],
    }
    entry.update(perf)
    return entry


def adjust_returns_for_direction(strategy_name: str, perf: Dict[str, float]) -> Dict[str, float]:
    """Invert forward returns for short setups so positive means the strategy worked."""
    if strategy_name not in SHORT_STRATEGIES:
        return perf
    adjusted = {}
    for key, value in perf.items():
        adjusted[key] = None if value is None else round(-value, 2)
    return adjusted


def backtest_regime(universe: List[str], days_ago: int = 30, cached_only: bool = True) -> Dict[str, List[Dict]]:
    fetcher = YahooFinanceFetcher()
    detector = load_regime_model()

    benchmark = fetcher.batch_download([BENCHMARK_TICKER], period="1y", cached_only=cached_only).get(BENCHMARK_TICKER, pd.DataFrame())
    if benchmark.empty:
        raise RuntimeError("Missing cached benchmark data for regime backtest")
    if isinstance(benchmark.columns, pd.MultiIndex):
        benchmark.columns = benchmark.columns.get_level_values(0)

    tickers = [ticker if "." in ticker else f"{ticker}.NS" for ticker in universe]
    all_data = fetcher.batch_download(tickers, period="1y", cached_only=cached_only)

    results: List[Dict] = []
    by_strategy = {
        "Momentum": [],
        "Mean Reversion": [],
        "Volatility Expansion": [],
        "Short / Defensive": [],
    }

    def analyze_ticker(ticker: str, df: pd.DataFrame):
        if df is None or df.empty or len(df) < max(80, days_ago + 30):
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        idx_signal = len(df) - days_ago - 1
        if idx_signal <= 60:
            return None

        hist_df = df.iloc[:idx_signal + 1]
        bench_hist = benchmark.loc[:hist_df.index[-1]]
        if len(bench_hist) < 60:
            return None

        regime_output = detector.predict(bench_hist)
        if regime_output.get("error"):
            return None

        strategy_weights = StrategyMetaController.get_allocation(regime_output["probabilities"])
        strategies = [
            MomentumStrategy(ticker, hist_df),
            MeanReversionStrategy(ticker, hist_df),
            VolatilityExpansionStrategy(ticker, hist_df),
            ShortDefensiveStrategy(ticker, hist_df),
        ]
        strategy_signals = [strategy.generate_signal() for strategy in strategies]
        alpha = SelectionEngine.calculate_alpha_score(ticker, strategy_signals, strategy_weights)
        if alpha["alpha_score"] <= 0.18 or not alpha.get("best_signal"):
            return None

        perf = calculate_returns(df, idx_signal, [1, 5, 10, 22, 30])
        perf = adjust_returns_for_direction(alpha["primary_strategy"], perf)
        return build_signal_entry(ticker, hist_df, alpha, perf, regime_output["regime"])

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(analyze_ticker, ticker, df): ticker for ticker, df in all_data.items()}
        for future in as_completed(futures):
            result = future.result()
            if not result:
                continue
            results.append(result)
            by_strategy[result["strategy"]].append(result)

    return {
        "all": results,
        **by_strategy,
    }


def summarize(label: str, results: List[Dict]) -> str:
    if not results:
        return f"\n{label}: No setups found.\n"
    df = pd.DataFrame(results)
    lines = [f"\n{'=' * 80}", f"  {label} ({len(df)} setups)", f"{'=' * 80}"]
    for period in [1, 5, 10, 22, 30]:
        col = f"{period}D"
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        lines.append(
            f"  {col:>4}: Avg {valid.mean():>+7.2f}%  Med {valid.median():>+7.2f}%  "
            f"WinRate {(valid > 0).mean() * 100:>5.1f}%  Best {valid.max():>+7.2f}%  Worst {valid.min():>+7.2f}%"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Backtest Regime Strategies")
    parser.add_argument("--index", type=str, default="NIFTY 500", help="Universe index")
    parser.add_argument("--full", action="store_true", help="Use full NSE universe")
    parser.add_argument("--days-ago", type=int, default=30, help="Days ago to generate setups")
    parser.add_argument("--test", action="store_true", help="Test mode")
    parser.add_argument("--short", action="store_true", help="Summary only")
    parser.add_argument("--cached-only", action="store_true", help="Use cached universe and prices only")
    args = parser.parse_args()

    uf = StockUniverseFetcher()
    tickers = (
        uf.fetch_universe(cached_only=args.cached_only)
        if args.full
        else uf.fetch_universe(index_name=args.index, cached_only=args.cached_only)
    )
    if args.test:
        tickers = tickers[:50]

    logger.info("Starting regime strategy backtest on %s symbols", len(tickers))
    results = backtest_regime(tickers, days_ago=args.days_ago, cached_only=args.cached_only)

    print(summarize("REGIME STRATEGY STACK", results["all"]))
    for label in ["Momentum", "Mean Reversion", "Volatility Expansion", "Short / Defensive"]:
        print(summarize(label, results[label]))

    if results["all"] and not args.short:
        print("\n" + "=" * 80)
        print("DETAILED REGIME BACKTEST")
        print("=" * 80)
        print(format_performance_summary(results["all"]))

    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if results["all"]:
        pd.DataFrame(results["all"]).to_csv(report_dir / f"backtest_regime_{ts}.csv", index=False)


if __name__ == "__main__":
    main()
