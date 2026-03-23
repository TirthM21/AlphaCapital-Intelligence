"""Stock Selection Engine (Layer 5) and Execution Rules (Layer 6) for Advanced Trading System.

Calculates Alpha-Scores and recommends position sizing/scaling rules.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class SelectionEngine:
    """Ranks and selects stocks using blended Alpha-Scores based on market regime."""

    @staticmethod
    def calculate_alpha_score(
        ticker: str,
        strategy_signals: List[Dict],
        strategy_weights: Dict[str, float]
    ) -> Dict:
        """Computes a single Alpha-Score for a stock by blending signals."""
        
        blended_score = 0.0
        best_strategy = None
        max_strength = -1.0
        
        details = {}
        for sig in strategy_signals:
            strategy_name = sig['strategy']
            weight = strategy_weights.get(strategy_name, 0.0)
            
            # Weighted contribution
            score_cont = sig['signal_strength'] * weight
            blended_score += score_cont
            
            details[strategy_name] = {
                'strength': sig['signal_strength'],
                'weight': weight,
                'contribution': score_cont
            }
            
            if sig['signal_strength'] > max_strength:
                max_strength = sig['signal_strength']
                best_strategy = sig
                
        return {
            'ticker': ticker,
            'alpha_score': round(blended_score, 4),
            'primary_strategy': best_strategy['strategy'] if best_strategy else None,
            'best_signal': best_strategy,
            'details': details
        }

class ExecutionEngine:
    """Handles position sizing and scaling rules (Layer 6)."""
    
    @staticmethod
    def calculate_position_size(
        price: float,
        stop_loss: float,
        portfolio_value: float,
        risk_per_trade_pct: float = 1.0, # Max risk 1% of equity
        max_pos_size_pct: float = 15.0 # Max 15% allocation
    ) -> Dict:
        """Calculates quantity based on risk amount (ATR-based stop/fixed stop)."""
        
        risk_amount = portfolio_value * (risk_per_trade_pct / 100)
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share <= 0:
            return {'error': 'Stop loss must be different from entry price'}
            
        # Quantity based on risk
        quantity_risk = int(risk_amount / risk_per_share)
        
        # Quantity based on max allocation
        max_allocation_amount = portfolio_value * (max_pos_size_pct / 100)
        quantity_max = int(max_allocation_amount / price)
        
        # Final quantity is the lower of the two
        final_qty = min(quantity_risk, quantity_max)
        
        return {
            'quantity': final_qty,
            'risk_per_share': round(risk_per_share, 2),
            'total_investment': round(final_qty * price, 2),
            'allocation_pct': round((final_qty * price / portfolio_value) * 100, 2),
            'actual_risk_pct': round((final_qty * risk_per_share / portfolio_value) * 100, 2)
        }

    @staticmethod
    def get_scaling_rules(df: pd.DataFrame, direction: str = 'bullish') -> List[str]:
        """Layer 6: Scaling IN/OUT rules."""
        rules = []
        
        # Scaling IN: 5-day EMA breakout
        ema_5 = df['Close'].ewm(span=5).mean().iloc[-1]
        curr_price = df['Close'].iloc[-1]
        
        if curr_price > ema_5:
            rules.append("Scale IN: Price above 5-day EMA")
            
        # Scaling OUT: Distribution volume (high volume on down day)
        avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
        is_down_day = df['Close'].iloc[-1] < df['Open'].iloc[-1]
        is_high_vol = df['Volume'].iloc[-1] > avg_vol * 1.5
        
        if is_down_day and is_high_vol:
            rules.append("Scale OUT: Distribution volume detected")
            
        return rules
