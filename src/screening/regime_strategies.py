"""Strategy Library (Layer 3) and Meta-Controller (Layer 4) for Advanced Trading System.

Calculates signal strength and performs dynamic allocation based on market regimes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

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
        
    def generate_signal(self) -> Dict:
        """Should return signal dict with entry, stop loss, and target."""
        pass

class MomentumStrategy(TradingStrategy):
    """Breakouts and Trending setups."""
    def generate_signal(self) -> Dict:
        # Detect breakout: Close > 52-week High (252 days)
        # Or simple N-day Donchian channel breakout
        lookback = 40
        high_40 = self.high.rolling(lookback).max().iloc[-2]
        curr_price = self.close.iloc[-1]
        
        signal_strength = 0
        if curr_price > high_40:
            signal_strength = 1.0 # Strong breakout
            
        # Entry, Stop Loss, Target
        entry = curr_price
        stop_loss = entry * 0.98 # 2% tight stop for momentum
        target = entry * 1.05 # 5% profit target
        
        return {
            'strategy': 'Momentum',
            'signal_strength': signal_strength,
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'expected_return': 5.0
        }

class MeanReversionStrategy(TradingStrategy):
    """Support bounces and RSI oversold setups."""
    def generate_signal(self) -> Dict:
        # Simple RSI based mean reversion
        from ..screening.indicators import calculate_rsi
        rsi = calculate_rsi(self.close, 14).iloc[-1]
        
        signal_strength = 0
        if rsi < 35:
            signal_strength = (35 - rsi) / 10 # Scale signal strength
            
        # Entry: Potential support at current low
        curr_price = self.close.iloc[-1]
        entry = curr_price
        stop_loss = entry * 0.97 # 3% stop
        target = entry * 1.03 # Mean reversion target (often 3% is enough)
        
        return {
            'strategy': 'Mean Reversion',
            'signal_strength': min(1.0, signal_strength),
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'expected_return': 3.0
        }

class VolatilityExpansionStrategy(TradingStrategy):
    """TTM Squeeze / Narrow Range breakouts."""
    def generate_signal(self) -> Dict:
        # Squeeze detection would be better
        # Simple: ATR ratio expansion
        atr = (self.high - self.low).rolling(14).mean()
        curr_atr = (self.high - self.low).iloc[-1]
        
        signal_strength = 0
        if curr_atr < atr.iloc[-1] * 0.8: # Very narrow range
            signal_strength = 0.5 # Compression phase
            
        curr_price = self.close.iloc[-1]
        entry = curr_price
        stop_loss = entry * 0.99 # Very tight for volatility expansion
        target = entry * 1.04
        
        return {
            'strategy': 'Volatility Expansion',
            'signal_strength': signal_strength,
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'expected_return': 4.0
        }

class ShortDefensiveStrategy(TradingStrategy):
    """Shorting weak stocks in bearish regimes."""
    def generate_signal(self) -> Dict:
        # Ticker crossing below 50-day SMA
        sma_50 = self.close.rolling(50).mean().iloc[-1]
        curr_price = self.close.iloc[-1]
        
        signal_strength = 0
        if curr_price < sma_50:
            signal_strength = 0.8
            
        entry = curr_price
        stop_loss = entry * 1.02 # Short stop is higher
        target = entry * 0.96 # Target lower
        
        return {
            'strategy': 'Short / Defensive',
            'signal_strength': signal_strength,
            'entry': round(entry, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'expected_return': 4.0
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
