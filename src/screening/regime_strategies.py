"""Strategy Library (Layer 3) and Meta-Controller (Layer 4) for Advanced Trading System.

Calculates signal strength and performs dynamic allocation based on market regimes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .indicators import calculate_rsi
from .phase_indicators import calculate_volume_ratio, classify_phase, detect_breakout, detect_vcp_pattern

logger = logging.getLogger(__name__)

class TradingStrategy:
    """Base class for all strategies."""
    def __init__(self, ticker: str, df: pd.DataFrame):
        self.ticker = ticker
        self.df = df
        self.close = df['Close']
        self.high = df['High']
        self.low = df['Low']
        self.vol = df['Volume']

    def _safe_volume_ratio(self, period: int = 20) -> float:
        if 'Volume' not in self.df.columns:
            return 1.0
        return float(calculate_volume_ratio(self.vol, period))

    def _phase_info(self) -> Dict:
        return classify_phase(self.df, float(self.close.iloc[-1]))

    def _bullish_reversal_bar(self) -> bool:
        if len(self.df) < 3:
            return False
        current = self.df.iloc[-1]
        previous = self.df.iloc[-2]
        body = abs(current["Close"] - current["Open"])
        candle_range = max(current["High"] - current["Low"], 1e-9)
        lower_wick = min(current["Open"], current["Close"]) - current["Low"]
        return (
            current["Close"] > current["Open"]
            and current["Close"] > previous["High"]
            and lower_wick / candle_range >= 0.25
            and body / candle_range >= 0.2
        )

    def _support_distance_pct(self, window: int = 20) -> float:
        recent_support = float(self.low.iloc[-window:].min())
        curr_price = float(self.close.iloc[-1])
        if recent_support <= 0:
            return 999.0
        return ((curr_price - recent_support) / recent_support) * 100.0

    def _breakout_context(self) -> Dict:
        curr_price = float(self.close.iloc[-1])
        phase_info = self._phase_info()
        vcp = detect_vcp_pattern(self.df, curr_price, phase_info)
        breakout = detect_breakout(self.df, curr_price, phase_info, vcp)
        return {
            "price": curr_price,
            "phase_info": phase_info,
            "vcp": vcp,
            "breakout": breakout,
            "volume_ratio": self._safe_volume_ratio(),
        }
        
    def generate_signal(self) -> Dict:
        """Should return signal dict with entry, stop loss, and target."""
        pass

class MomentumStrategy(TradingStrategy):
    """Breakouts and Trending setups."""
    def generate_signal(self) -> Dict:
        if len(self.df) < 220:
            return {
                'strategy': 'Momentum',
                'signal_strength': 0.0,
                'entry': 0.0,
                'stop_loss': 0.0,
                'target': 0.0,
                'expected_return': 0.0
            }

        ctx = self._breakout_context()
        curr_price = ctx["price"]
        phase_info = ctx["phase_info"]
        breakout = ctx["breakout"]
        volume_ratio = ctx["volume_ratio"]
        vcp = ctx["vcp"]

        momentum_ratio = self.close / self.close.rolling(50).mean()
        rs_slope = float(momentum_ratio.iloc[-1] - momentum_ratio.iloc[-6]) if len(momentum_ratio.dropna()) >= 6 else 0.0
        distance_52w = phase_info.get("week_52_high", curr_price)
        distance_52w = ((distance_52w - curr_price) / distance_52w * 100) if distance_52w else 100.0

        above_sma_stack = (
            curr_price > phase_info.get("sma_50", curr_price)
            and phase_info.get("sma_50", 0) > phase_info.get("sma_200", 0)
        )
        not_extended = 2 <= phase_info.get("distance_from_50sma", 0) <= 18
        rs_ok = rs_slope > 0.02
        volume_ok = volume_ratio >= 1.2 or breakout.get("volume_confirmed", False)
        breakout_ok = breakout.get("is_breakout") or (vcp.get("is_vcp") and vcp.get("vcp_quality", 0) >= 65)
        near_high = distance_52w <= 12

        signal_strength = 0.0
        if (
            phase_info.get("phase") == 2
            and above_sma_stack
            and breakout_ok
            and rs_ok
            and volume_ok
            and near_high
            and not_extended
        ):
            signal_strength = min(
                1.0,
                0.35
                + min(0.2, max(0.0, rs_slope * 2.5))
                + min(0.15, max(0.0, (volume_ratio - 1.0) * 0.25))
                + min(0.15, vcp.get("vcp_quality", 0) / 500.0)
                + min(0.15, max(0.0, (12 - distance_52w) / 12.0) * 0.15),
            )

        entry = curr_price
        stop_anchor = min(float(self.low.iloc[-10:].min()), phase_info.get("sma_50", curr_price))
        stop_loss = min(entry * 0.95, stop_anchor * 0.995) if stop_anchor > 0 else entry * 0.95
        target = entry * 1.10

        return {
            'strategy': 'Momentum',
            'signal_strength': round(signal_strength, 4),
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'expected_return': 10.0
        }

class MeanReversionStrategy(TradingStrategy):
    """Support bounces and RSI oversold setups."""
    def generate_signal(self) -> Dict:
        if len(self.df) < 220:
            return {
                'strategy': 'Mean Reversion',
                'signal_strength': 0.0,
                'entry': 0.0,
                'stop_loss': 0.0,
                'target': 0.0,
                'expected_return': 0.0
            }

        rsi = float(calculate_rsi(self.close, 14).iloc[-1])
        phase_info = self._phase_info()
        curr_price = float(self.close.iloc[-1])
        support_distance = self._support_distance_pct()
        volume_ratio = self._safe_volume_ratio()
        reversal_bar = self._bullish_reversal_bar()
        bounce_confirmed = len(self.close) >= 3 and curr_price > float(self.close.iloc[-2]) > float(self.close.iloc[-3])
        trend_not_broken = curr_price >= phase_info.get("sma_200", curr_price * 0.95) * 0.96
        recent_high = float(self.high.iloc[-21:-1].max())
        pullback_pct = ((recent_high - curr_price) / recent_high) * 100 if recent_high > 0 else 0.0

        signal_strength = 0.0
        if (
            rsi <= 50
            and support_distance <= 12.0
            and pullback_pct >= 5.0
            and reversal_bar
            and bounce_confirmed
            and trend_not_broken
            and phase_info.get("phase") in [1, 2]
            and volume_ratio >= 0.9
        ):
            signal_strength = min(
                0.9,
                0.30
                + min(0.20, max(0.0, (50 - rsi) / 25.0))
                + min(0.20, max(0.0, (12.0 - support_distance) / 12.0) * 0.20)
                + min(0.15, max(0.0, pullback_pct / 12.0) * 0.15)
                + (0.15 if bounce_confirmed else 0.0),
            )

        entry = curr_price
        stop_loss = float(self.low.iloc[-5:].min()) * 0.995
        target = max(entry * 1.05, phase_info.get("sma_50", entry))

        return {
            'strategy': 'Mean Reversion',
            'signal_strength': round(min(1.0, signal_strength), 4),
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'expected_return': 5.0
        }

class VolatilityExpansionStrategy(TradingStrategy):
    """TTM Squeeze / Narrow Range breakouts."""
    def generate_signal(self) -> Dict:
        if len(self.df) < 220:
            return {
                'strategy': 'Volatility Expansion',
                'signal_strength': 0.0,
                'entry': 0.0,
                'stop_loss': 0.0,
                'target': 0.0,
                'expected_return': 0.0
            }

        ctx = self._breakout_context()
        curr_price = ctx["price"]
        phase_info = ctx["phase_info"]
        breakout = ctx["breakout"]
        vcp = ctx["vcp"]
        volume_ratio = ctx["volume_ratio"]

        tr = self.high - self.low
        atr = tr.rolling(14).mean()
        current_atr = float(atr.iloc[-1]) if not atr.empty else 0.0
        prior_atr = float(atr.iloc[-11:-1].mean()) if len(atr.dropna()) >= 20 else current_atr
        atr_expanding = current_atr > prior_atr * 1.15 if prior_atr > 0 else False
        range_20_close_high = float(self.close.iloc[-21:-1].max())
        range_breakout = curr_price > range_20_close_high
        recent_range = float(tr.iloc[-20:-5].mean()) if len(tr) >= 30 else current_atr
        earlier_range = float(tr.iloc[-60:-20].mean()) if len(tr) >= 80 else prior_atr
        pre_breakout_contraction = bool(vcp.get("is_vcp")) or (
            earlier_range > 0 and recent_range < earlier_range * 0.75
        )

        signal_strength = 0.0
        if (
            phase_info.get("phase") in [1, 2]
            and pre_breakout_contraction
            and breakout.get("is_breakout")
            and range_breakout
            and atr_expanding
            and volume_ratio >= 1.3
        ):
            signal_strength = min(
                1.0,
                0.35
                + min(0.25, max(0.0, (volume_ratio - 1.0) * 0.3))
                + (0.20 if vcp.get("is_vcp") else 0.10)
                + (0.15 if atr_expanding else 0.0),
            )

        entry = curr_price
        stop_loss = min(entry * 0.96, float(self.low.iloc[-10:].min()) * 0.995)
        target = entry * 1.08

        return {
            'strategy': 'Volatility Expansion',
            'signal_strength': round(signal_strength, 4),
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'expected_return': 8.0
        }

class ShortDefensiveStrategy(TradingStrategy):
    """Shorting weak stocks in bearish regimes."""
    def generate_signal(self) -> Dict:
        if len(self.df) < 220:
            return {
                'strategy': 'Short / Defensive',
                'signal_strength': 0.0,
                'entry': 0.0,
                'stop_loss': 0.0,
                'target': 0.0,
                'expected_return': 0.0
            }

        phase_info = self._phase_info()
        curr_price = float(self.close.iloc[-1])
        sma_50 = float(self.close.rolling(50).mean().iloc[-1])
        sma_200 = float(self.close.rolling(200).mean().iloc[-1])
        volume_ratio = self._safe_volume_ratio()

        lower_high = float(self.high.iloc[-1]) < float(self.high.iloc[-10:-1].max())
        lower_low = float(self.low.iloc[-1]) <= float(self.low.iloc[-10:-1].min())
        weak_trend_stack = curr_price < sma_50 < sma_200
        downside_extension = phase_info.get("distance_from_50sma", 0) <= -3
        slopes_negative = phase_info.get("slope_50", 0) < 0 and phase_info.get("slope_200", 0) <= 0

        signal_strength = 0.0
        if (
            phase_info.get("phase") == 4
            and weak_trend_stack
            and lower_high
            and lower_low
            and downside_extension
            and slopes_negative
            and volume_ratio >= 1.0
        ):
            signal_strength = min(
                1.0,
                0.40
                + min(0.20, max(0.0, abs(phase_info.get("distance_from_50sma", 0)) / 15.0) * 0.20)
                + (0.15 if lower_high else 0.0)
                + (0.15 if lower_low else 0.0)
                + min(0.10, max(0.0, (volume_ratio - 1.0) * 0.1)),
            )

        entry = curr_price
        stop_loss = max(entry * 1.05, float(self.high.iloc[-5:].max()) * 1.005)
        target = entry * 0.90

        return {
            'strategy': 'Short / Defensive',
            'signal_strength': round(signal_strength, 4),
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'expected_return': 10.0
        }

class StrategyMetaController:
    """Allocates weights to each strategy based on current market regime."""
    
    # Static allocation map based on regime names from MarketRegimeDetector
    ALLOCATION_MAP = {
        "Bullish Trending": {
            "Momentum": 0.7,
            "Mean Reversion": 0.2,
            "Volatility Expansion": 0.1,
            "Short / Defensive": 0.0
        },
        "Sideways / Mean Reverting": {
            "Momentum": 0.1,
            "Mean Reversion": 0.7,
            "Volatility Expansion": 0.2,
            "Short / Defensive": 0.0
        },
        "Volatile / Defensive": {
            "Momentum": 0.0,
            "Mean Reversion": 0.3,
            "Volatility Expansion": 0.2,
            "Short / Defensive": 0.5
        },
        "Crisis / High Vol": {
            "Momentum": 0.0,
            "Mean Reversion": 0.1,
            "Volatility Expansion": 0.1,
            "Short / Defensive": 0.8
        }
    }

    @staticmethod
    def get_allocation(regime_probs: Dict[str, float]) -> Dict[str, float]:
        """Calculates blended strategy weights based on regime probabilities."""
        weights = {
            "Momentum": 0.0,
            "Mean Reversion": 0.0,
            "Volatility Expansion": 0.0,
            "Short / Defensive": 0.0
        }
        
        for regime_name, prob in regime_probs.items():
            regime_weights = StrategyMetaController.ALLOCATION_MAP.get(regime_name, {})
            for strategy, weight in regime_weights.items():
                weights[strategy] += weight * prob
                
        return weights
