"""Advanced Backtesting Engine for phase-based trading strategies.

Simulates historical portfolio performance, drawdown, and win rates.
Compares strategy against Nifty 50 benchmark.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.screening.phase_indicators import (
    calculate_relative_strength,
    calculate_rs_slope,
    classify_phase,
    detect_vcp_pattern,
    validate_minervini_trend_template,
)

logger = logging.getLogger(__name__)


class AdvancedBacktester:
    """Simulates trading strategies on historical data."""

    def __init__(self, initial_capital: float = 1000000.0, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.cash = initial_capital
        self.positions = {}
        self.trade_history = []
        self.benchmark_ticker = "^NSEI"

    def _reset_state(self) -> None:
        """Reset internal state so one instance can safely run multiple backtests."""
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_history = []

    def _score_entry_candidate(
        self,
        ticker: str,
        hist: pd.DataFrame,
        bench_hist: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Rank candidates so the strategy buys the strongest setups first."""
        price = float(hist.iloc[-1]["Close"])
        phase_info = classify_phase(hist, price)
        sma_200_series = hist["Close"].rolling(window=200).mean()
        minervini = validate_minervini_trend_template(price, phase_info, sma_200_series)

        if phase_info["phase"] != 2 or not minervini["passes_template"]:
            return {"ticker": ticker, "is_candidate": False}

        rs_series = calculate_relative_strength(hist["Close"], bench_hist["Close"])
        rs_slope = calculate_rs_slope(rs_series, 20) if not rs_series.empty else 0.0
        vcp = detect_vcp_pattern(hist, price, phase_info)

        volume_ratio = 1.0
        if "Volume" in hist.columns and len(hist) >= 21:
            avg_volume = hist["Volume"].iloc[-21:-1].mean()
            if avg_volume and not np.isnan(avg_volume):
                volume_ratio = float(hist["Volume"].iloc[-1] / avg_volume)

        trend_strength = max(0.0, phase_info.get("distance_from_50sma", 0.0))
        slope_strength = max(0.0, phase_info.get("slope_50", 0.0) * 100)
        proximity_bonus = max(
            0.0,
            25 - minervini["criteria_details"].get("distance_from_52w_high_pct", 25),
        )
        vcp_bonus = vcp.get("vcp_quality", 0.0) / 5 if vcp.get("is_vcp") else 0.0
        volume_bonus = min(10.0, max(0.0, (volume_ratio - 1.0) * 10))

        score = (
            minervini["template_score"]
            + (rs_slope * 40)
            + trend_strength
            + slope_strength
            + proximity_bonus
            + vcp_bonus
            + volume_bonus
        )

        breakout_ready = (
            2.0 <= phase_info.get("distance_from_50sma", 0.0) <= 18.0
            and rs_slope > 0.025
            and minervini["template_score"] >= 87
            and (
                volume_ratio >= 1.2
                or (vcp.get("is_vcp") and vcp.get("vcp_quality", 0.0) >= 65)
            )
        )
        if not breakout_ready:
            return {"ticker": ticker, "is_candidate": False}

        return {
            "ticker": ticker,
            "is_candidate": True,
            "score": round(score, 2),
            "price": price,
            "rs_slope": round(rs_slope, 4),
            "vcp_quality": round(vcp.get("vcp_quality", 0.0), 1),
        }

    def _load_benchmark_data(
        self,
        fetcher,
        start_date: str,
        end_date: str,
        period: str,
        cached_only: bool = False,
    ) -> pd.DataFrame:
        """Load benchmark data via shared fetcher cache before attempting network."""
        bench_df = fetcher.batch_download(
            [self.benchmark_ticker],
            period=period,
            cached_only=cached_only,
        ).get(self.benchmark_ticker)
        if bench_df is None or bench_df.empty:
            cache_candidates = sorted(
                fetcher.cache_dir.glob(f"{self.benchmark_ticker}_prices_*_1d.pkl"),
                key=lambda path: path.stat().st_size,
                reverse=True,
            )
            for cache_path in cache_candidates:
                try:
                    cached_df = pd.read_pickle(cache_path)
                except Exception:
                    continue
                if isinstance(cached_df, pd.DataFrame) and not cached_df.empty:
                    bench_df = cached_df
                    logger.warning("Using benchmark cache fallback from %s", cache_path.name)
                    break

        if bench_df is None or bench_df.empty:
            return pd.DataFrame()

        if isinstance(bench_df.columns, pd.MultiIndex):
            bench_df.columns = bench_df.columns.get_level_values(0)
        return bench_df.loc[start_date:end_date]

    def run_backtest(self, symbols: List[str], start_date: str, end_date: str, cached_only: bool = False) -> Dict[str, Any]:
        """Run full historical simulation."""
        self._reset_state()
        if not symbols:
            return {"error": "No symbols available for backtest"}
        logger.info(
            "Starting backtest from %s to %s for %s stocks...",
            start_date,
            end_date,
            len(symbols),
        )

        from src.data.fetcher import YahooFinanceFetcher

        fetcher = YahooFinanceFetcher()

        days_diff = (
            datetime.strptime(end_date, "%Y-%m-%d")
            - datetime.strptime(start_date, "%Y-%m-%d")
        ).days
        if days_diff <= 365:
            period = "1y"
        elif days_diff <= 730:
            period = "2y"
        elif days_diff <= 1825:
            period = "5y"
        elif days_diff <= 3650:
            period = "10y"
        else:
            period = "max"

        logger.info("Batch downloading %s stocks with period=%s...", len(symbols), period)
        all_data = fetcher.batch_download(symbols, period=period, cached_only=cached_only)

        for symbol in list(all_data.keys()):
            df = all_data[symbol]
            all_data[symbol] = df.loc[start_date:end_date]
            if all_data[symbol].empty:
                del all_data[symbol]

        logger.info("Data loaded for %s stocks.", len(all_data))
        if not all_data:
            return {"error": "No historical price data available for backtest"}

        bench_df = self._load_benchmark_data(fetcher, start_date, end_date, period, cached_only=cached_only)
        if bench_df.empty:
            return {"error": "Benchmark download failed"}

        dates = bench_df.index
        daily_portfolio_values = []

        for i in range(200, len(dates)):
            curr_date = dates[i]

            current_portfolio_value = self.cash
            for ticker, pos in list(self.positions.items()):
                hist_to_date = all_data[ticker].loc[:curr_date]
                if hist_to_date.empty:
                    continue

                day_price = float(hist_to_date.iloc[-1]["Close"])
                current_portfolio_value += pos["qty"] * day_price
                pos["max_price"] = max(pos["max_price"], day_price)

                gain = (day_price - pos["entry_price"]) / pos["entry_price"]
                peak_gain = (pos["max_price"] - pos["entry_price"]) / pos["entry_price"]

                exit_signal = False
                if gain <= -0.06:
                    exit_signal = True
                elif peak_gain > 0.10 and gain < 0.03:
                    exit_signal = True
                elif peak_gain > 0.15 and day_price < pos["max_price"] * 0.90:
                    exit_signal = True
                elif gain >= 0.25:
                    exit_signal = True

                if exit_signal:
                    self.cash += pos["qty"] * day_price
                    self.trade_history.append(
                        {
                            "ticker": ticker,
                            "qty": pos["qty"],
                            "entry": pos["entry_price"],
                            "exit": day_price,
                            "gain": float(gain * 100),
                            "duration": (curr_date - pos["entry_date"]).days,
                        }
                    )
                    del self.positions[ticker]

            daily_portfolio_values.append(
                {
                    "date": curr_date,
                    "value": current_portfolio_value,
                    "bench_price": bench_df.loc[curr_date, "Close"],
                }
            )

            bench_hist = bench_df.loc[:curr_date]
            bench_info = classify_phase(bench_hist, float(bench_hist.iloc[-1]["Close"]))
            is_market_healthy = bench_info["phase"] == 2

            if is_market_healthy and len(self.positions) < self.max_positions:
                candidates = []
                for ticker, df in all_data.items():
                    if ticker in self.positions:
                        continue

                    hist = df.loc[:curr_date]
                    if len(hist) < 200:
                        continue

                    candidate = self._score_entry_candidate(ticker, hist, bench_hist)
                    if candidate.get("is_candidate"):
                        candidates.append(candidate)

                candidates.sort(key=lambda item: item["score"], reverse=True)

                slots_left = self.max_positions - len(self.positions)
                for candidate in candidates[:slots_left]:
                    price = float(candidate["price"])
                    remaining_slots = self.max_positions - len(self.positions)
                    allocation = min(
                        self.cash / max(1, remaining_slots),
                        self.initial_capital * 0.10,
                    )
                    qty = int(allocation // price)
                    if qty <= 0:
                        continue

                    self.positions[candidate["ticker"]] = {
                        "qty": qty,
                        "entry_price": price,
                        "max_price": price,
                        "entry_date": curr_date,
                        "entry_score": candidate["score"],
                        "rs_slope": candidate["rs_slope"],
                        "vcp_quality": candidate["vcp_quality"],
                    }
                    self.cash -= qty * price

        final_df = pd.DataFrame(daily_portfolio_values)
        if final_df.empty:
            return {"error": "No trades executed"}

        final_df["cum_return"] = (final_df["value"] / self.initial_capital - 1) * 100
        final_df["bench_return"] = (
            final_df["bench_price"] / final_df["bench_price"].iloc[0] - 1
        ) * 100
        final_df["peak"] = final_df["value"].cummax()
        final_df["drawdown"] = (final_df["value"] - final_df["peak"]) / final_df["peak"]
        max_dd = final_df["drawdown"].min() * 100

        wins = [trade for trade in self.trade_history if trade["gain"] > 0]
        losses = [trade for trade in self.trade_history if trade["gain"] <= 0]
        win_rate = (len(wins) / len(self.trade_history) * 100) if self.trade_history else 0

        gain_val = sum(trade["gain"] for trade in wins)
        loss_val = abs(sum(trade["gain"] for trade in losses))
        profit_factor = round(gain_val / loss_val, 2) if loss_val != 0 else float("inf")

        years = max(
            (final_df["date"].iloc[-1] - final_df["date"].iloc[0]).days / 365.25,
            1 / 365.25,
        )
        ending_value = float(final_df["value"].iloc[-1])
        cagr = ((ending_value / self.initial_capital) ** (1 / years) - 1) * 100
        avg_trade = float(np.mean([trade["gain"] for trade in self.trade_history])) if self.trade_history else 0.0

        return {
            "total_return": round(final_df["cum_return"].iloc[-1], 2),
            "benchmark_return": round(final_df["bench_return"].iloc[-1], 2),
            "cagr": round(cagr, 2),
            "win_rate": round(win_rate, 2),
            "avg_trade_gain": round(avg_trade, 2),
            "profit_factor": profit_factor,
            "max_drawdown": round(max_dd, 2),
            "total_trades": len(self.trade_history),
            "daily_values": final_df,
        }

    def format_summary(self, results: Dict[str, Any]) -> str:
        """Format metrics for the user."""
        if "error" in results:
            return results["error"]

        lines = [
            "=" * 80,
            "STRATEGY PERFORMANCE BACKTEST (PHASE 2 / TREND TEMPLATE)",
            "=" * 80,
            f"{'Total Return:':<25} {results['total_return']:>10}%",
            f"{'Benchmark (Nifty 50):':<25} {results['benchmark_return']:>10}%",
            f"{'Alpha:':<25} {results['total_return'] - results['benchmark_return']:>10.2f}%",
            f"{'CAGR:':<25} {results['cagr']:>10}%",
            "-" * 80,
            f"{'Win Rate:':<25} {results['win_rate']:>10}%",
            f"{'Avg Trade Gain:':<25} {results['avg_trade_gain']:>10}%",
            f"{'Profit Factor:':<25} {results['profit_factor']:>10}",
            f"{'Max Drawdown:':<25} {results['max_drawdown']:>10}%",
            f"{'Total Trades:':<25} {results['total_trades']:>10}",
            "=" * 80,
        ]

        summary = "\n".join(lines)
        logger.info("\n%s", summary)
        return summary
