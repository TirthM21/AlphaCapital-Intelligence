"""Advanced Backtesting Engine for phase-based trading strategies.

Simulates historical portfolio performance, drawdown, and win rates.
Compares strategy against Nifty 50 benchmark.
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.screening.phase_indicators import classify_phase, validate_minervini_trend_template
from src.screening.extended_signals import score_extended_signals

logger = logging.getLogger(__name__)

class AdvancedBacktester:
    """Simulates trading strategies on historical data."""

    def __init__(self, initial_capital: float = 1000000.0, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.cash = initial_capital
        self.positions = {} # {ticker: {'qty': N, 'entry_price': P, 'entry_date': D}}
        self.trade_history = []
        self.benchmark_ticker = "^NSEI"

    def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, any]:
        """Run full historical simulation."""
        logger.info(f"Starting backtest from {start_date} to {end_date} for {len(symbols)} stocks...")
        
        # 1. Fetch data for all symbols + benchmark
        all_data = {}
        for s in symbols:
            df = yf.download(s, start=start_date, end=end_date, progress=False)
            if not df.empty:
                # Standardize columns
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                all_data[s] = df
        
        bench_df = yf.download(self.benchmark_ticker, start=start_date, end=end_date, progress=False)
        if isinstance(bench_df.columns, pd.MultiIndex): bench_df.columns = bench_df.columns.get_level_values(0)

        # 2. Daily simulation loop
        dates = bench_df.index
        daily_portfolio_values = []
        
        for i in range(200, len(dates)): # Start after 200 days to allow SMA calculations
            curr_date = dates[i]
            
            # Update current value of open positions
            current_portfolio_value = self.cash
            for ticker, pos in list(self.positions.items()):
                day_price = float(all_data[ticker].loc[:curr_date].iloc[-1]['Close'])
                current_portfolio_value += pos['qty'] * day_price
                
                # Update Max Price for Trailing Stop
                pos['max_price'] = max(pos['max_price'], day_price)
                
                # Realistic Risk/Exit Logic:
                gain = (day_price - pos['entry_price']) / pos['entry_price']
                peak_gain = (pos['max_price'] - pos['entry_price']) / pos['entry_price']
                
                exit_signal = False
                
                # 1. Trailing Stop: If up 10%, move stop to Breakeven
                if peak_gain > 0.10 and gain < 0.01:
                    exit_signal = True # Sold at Breakeven
                    
                # 2. Hard Stop Loss: 8%
                elif gain <= -0.08:
                    exit_signal = True
                
                # 3. Dynamic Profit Target: Sell at 25% OR if price drops 10% from peak
                elif gain >= 0.25 or (peak_gain > 0.15 and day_price < pos['max_price'] * 0.90):
                    exit_signal = True

                if exit_signal:
                    self.cash += pos['qty'] * day_price
                    self.trade_history.append({
                        'ticker': ticker,
                        'qty': pos['qty'],
                        'entry': pos['entry_price'],
                        'exit': day_price,
                        'gain': float(gain * 100),
                        'duration': (curr_date - pos['entry_date']).days
                    })
                    del self.positions[ticker]

            daily_portfolio_values.append({
                'date': curr_date,
                'value': current_portfolio_value,
                'bench_price': bench_df.loc[curr_date, 'Close']
            })

            # Market Regime Filter: Only buy if benchmark is HEALTHY (Phase 1 or 2)
            # Use data up to the PREVIOUS day to avoid lookahead bias
            bench_hist = bench_df.loc[:curr_date]
            bench_info = classify_phase(bench_hist, bench_hist.iloc[-1]['Close'])
            is_market_healthy = bench_info['phase'] in [1, 2]

            # Check for NEW BUY signals if market is healthy and we have room
            if is_market_healthy and len(self.positions) < self.max_positions:
                for ticker, df in all_data.items():
                    if ticker in self.positions: continue
                    
                    # Look at data up to TODAY
                    hist = df.loc[:curr_date]
                    if len(hist) < 200: continue
                    
                    # Strategy: Phase 2 + Trend Template
                    price = float(hist.iloc[-1]['Close'])
                    phase_info = classify_phase(hist, price)
                    
                    # Calculate SMA 200 for the Trend Template
                    sma_200_series = hist['Close'].rolling(window=200).mean()
                    minervini_results = validate_minervini_trend_template(price, phase_info, sma_200_series)
                    
                    if phase_info['phase'] == 2 and minervini_results['passes_template']:
                        # Buy Signal
                        allocation = (self.initial_capital * 0.1) # 10% per trade
                        if self.cash >= allocation:
                            qty = int(allocation // price)
                            self.positions[ticker] = {
                                'qty': qty,
                                'entry_price': price,
                                'max_price': price, # Use for trailing
                                'entry_date': curr_date
                            }
                            self.cash -= (qty * price)

        # 3. Calculate Final Metrics
        final_df = pd.DataFrame(daily_portfolio_values)
        if final_df.empty: return {"error": "No trades executed"}
        
        # Portfolio Stats
        final_df['cum_return'] = (final_df['value'] / self.initial_capital - 1) * 100
        final_df['bench_return'] = (final_df['bench_price'] / final_df['bench_price'].iloc[0] - 1) * 100
        
        # Drawdown
        final_df['peak'] = final_df['value'].cummax()
        final_df['drawdown'] = (final_df['value'] - final_df['peak']) / final_df['peak']
        max_dd = final_df['drawdown'].min() * 100
        
        # Trade Stats
        wins = [t for t in self.trade_history if t['gain'] > 0]
        losses = [t for t in self.trade_history if t['gain'] <= 0]
        win_rate = (len(wins) / len(self.trade_history) * 100) if self.trade_history else 0
        
        total_profit = sum([t['exit'] * t['qty'] - t['entry'] * t['qty'] for t in self.trade_history if t['gain'] > 0]) # Simplified
        # Re-calculating profit factor from gains
        gain_val = sum([t['gain'] for t in wins])
        loss_val = abs(sum([t['gain'] for t in losses]))
        profit_factor = round(gain_val / loss_val, 2) if loss_val != 0 else float('inf')

        return {
            'total_return': round(final_df['cum_return'].iloc[-1], 2),
            'benchmark_return': round(final_df['bench_return'].iloc[-1], 2),
            'win_rate': round(win_rate, 2),
            'profit_factor': profit_factor,
            'max_drawdown': round(max_dd, 2),
            'total_trades': len(self.trade_history),
            'daily_values': final_df
        }

    def format_summary(self, results: Dict) -> str:
        """Format metrics for the user."""
        if "error" in results: return results['error']
        
        lines = []
        lines.append("="*80)
        lines.append("STRATEGY PERFORMANCE BACKTEST (PHASE 2 / TREND TEMPLATE)")
        lines.append("="*80)
        lines.append(f"{'Total Return:':<25} {results['total_return']:>10}%")
        lines.append(f"{'Benchmark (Nifty 50):':<25} {results['benchmark_return']:>10}%")
        lines.append(f"{'Alpha:':<25} {results['total_return'] - results['benchmark_return']:>10.2f}%")
        lines.append("-" * 80)
        lines.append(f"{'Win Rate:':<25} {results['win_rate']:>10}%")
        lines.append(f"{'Profit Factor:':<25} {results['profit_factor']:>10}")
        lines.append(f"{'Max Drawdown:':<25} {results['max_drawdown']:>10}%")
        lines.append(f"{'Total Trades:':<25} {results['total_trades']:>10}")
        lines.append("="*80)
        
        summary = "\n".join(lines)
        logger.info(f"\n{summary}")
        return summary
