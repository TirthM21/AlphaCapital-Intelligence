#!/usr/bin/env python3
"""Main orchestrator for the Advanced Regime-Switching Trading System.

Integrates Layers 1-7 for a fully adaptive, multi-strategy system.
"""

import argparse
import logging
import sys
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.universe_fetcher import StockUniverseFetcher
from src.data.fetcher import YahooFinanceFetcher
from src.analysis.regime_model import MarketRegimeDetector
from src.screening.regime_strategies import (
    MomentumStrategy, MeanReversionStrategy, 
    VolatilityExpansionStrategy, ShortDefensiveStrategy,
    StrategyMetaController
)
from src.screening.selection_engine import SelectionEngine, ExecutionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BENCHMARK_TICKERS = ["^NSEI", "^NSEBANK", "NIFTY_FIN_SERVICE.NS", "NIFTY_MIDCAP_100.NS"]
MODEL_PATH = "data/cache/regime_model.pkl"

def setup_directories():
    """Ensure required directories exist."""
    dirs = ["data/cache", "data/reports"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def update_regime_model(fetcher: YahooFinanceFetcher) -> MarketRegimeDetector:
    """Train or load the regime model (Layer 2)."""
    detector = MarketRegimeDetector(n_regimes=4, lookback=252*5) # 5 years
    
    # Try to load existing
    if os.path.exists(MODEL_PATH):
        try:
            detector = MarketRegimeDetector.load_state(MODEL_PATH)
            logger.info("Loaded existing regime model.")
            # We skip retraining for now for speed, but Layer 7 suggests rolling retraining
            # return detector 
        except Exception:
            logger.warning("Failed to load model, retraining...")
            
    # Retrain (Layer 7: Adaptation)
    logger.info(f"Downloading benchmark data for {BENCHMARK_TICKERS}...")
    bench_data = fetcher.batch_download(BENCHMARK_TICKERS, period="5y")
    
    valid_benchmarks = [d for d in bench_data.values() if not d.empty and len(d) > 250]
    
    if len(valid_benchmarks) < 2:
        logger.error("Insufficient benchmark data for retraining.")
        return detector
        
    detector.train(valid_benchmarks)
    detector.save_state(MODEL_PATH)
    return detector

def run_regime_scan(universe: List[str], portfolio_value: float = 1000000.0):
    """Orchestrate the advanced scan (Layers 3-6)."""
    fetcher = YahooFinanceFetcher()
    setup_directories()
    
    # 1. Regime Detection (Layer 1 & 2)
    # Use ^NSEI as primary regime driver
    detector = update_regime_model(fetcher)
    nifty_df = fetcher.fetch_price_history("^NSEI", period="1y")
    
    if nifty_df.empty:
        logger.error("Could not fetch NIFTY 50 data for regime prediction")
        return
        
    regime_output = detector.predict(nifty_df)
    current_regime = regime_output['regime']
    probs = regime_output['probabilities']
    
    logger.info("="*80)
    logger.info(f"CURRENT MARKET REGIME: {current_regime.upper()}")
    logger.info(f"Probabilities: {probs}")
    logger.info("="*80)
    
    # 2. Meta-Controller: Strategy Allocation (Layer 4)
    strategy_weights = StrategyMetaController.get_allocation(probs)
    logger.info(f"Dynamic Strategy Allocation: {strategy_weights}")
    
    # 3. Data Fetching for Universe
    logger.info(f"Fetching data for {len(universe)} stocks...")
    # Use 300 stocks chunks for speed
    chunk_size = 300
    all_results = []
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def analyze_stock(ticker, df):
        if df.empty or len(df) < 50:
            return None
            
        # Standardize columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Generate Signals from Library (Layer 3)
        strategies = [
            MomentumStrategy(ticker, df),
            MeanReversionStrategy(ticker, df),
            VolatilityExpansionStrategy(ticker, df),
            ShortDefensiveStrategy(ticker, df)
        ]
        
        signals = []
        for strat in strategies:
            signals.append(strat.generate_signal())
            
        # Stock Selection: Alpha-Score (Layer 5)
        alpha = SelectionEngine.calculate_alpha_score(ticker, signals, strategy_weights)
        
        if alpha['alpha_score'] > 0.1: # Threshold for recommendation
            # Execution: Position Sizing (Layer 6)
            best_sig = alpha['best_signal']
            sizing = ExecutionEngine.calculate_position_size(
                price=best_sig['entry'],
                stop_loss=best_sig['stop_loss'],
                portfolio_value=portfolio_value,
                risk_per_trade_pct=1.0 # 1% risk per trade
            )
            
            # Scaling Rules
            scaling = ExecutionEngine.get_scaling_rules(df)
            
            result = {
                'ticker': ticker,
                'score': alpha['alpha_score'],
                'strategy': alpha['primary_strategy'],
                'price': best_sig['entry'],
                'stop': best_sig['stop_loss'],
                'target': best_sig['target'],
                'expected_return': best_sig['expected_return'],
                'quantity': sizing.get('quantity', 0),
                'allocation_pct': sizing.get('allocation_pct', 0),
                'risk_pct': sizing.get('actual_risk_pct', 0),
                'scaling_rules': ", ".join(scaling)
            }
            return result
        return None

    for i in range(0, len(universe), chunk_size):
        chunk = universe[i:i+chunk_size]
        logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(universe)-1)//chunk_size + 1}...")
        
        chunk_data = fetcher.batch_download(chunk, period="1y")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(analyze_stock, t, d): t 
                for t, d in chunk_data.items()
            }
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    all_results.append(res)
                    
    # 4. Generate Report
    if not all_results:
        logger.info("No actionable signals found for current allocation.")
        return
        
    df_results = pd.DataFrame(all_results).sort_values(by='score', ascending=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"data/reports/regime_scan_{timestamp}.csv"
    df_results.to_csv(report_file, index=False)
    
    # Console Output
    print("\n" + "="*100)
    print(f"REGIME-SWITCHING SELECTIONS (Market: {current_regime})")
    print("="*100)
    print(df_results.head(20).to_string(index=False))
    print("="*100)
    print(f"Full report saved to: {report_file}")
    
    # 5. Telegram Notification
    from src.notifications.telegram_notifier import TelegramNotifier
    telegram = TelegramNotifier()
    telegram.send_signals(all_results, f"Extended Markov Scan ({current_regime})")

def main():
    parser = argparse.ArgumentParser(description='Advanced Regime-Switching Scanner')
    parser.add_argument('--index', type=str, default='NIFTY 50', help='Index to scan')
    parser.add_argument('--test', action='store_true', help='Test with 20 stocks')
    parser.add_argument('--full', action='store_true', help='Scan NIFTY 500')
    parser.add_argument('--short', action='store_true', help='Summarize output (<100 lines)')
    args = parser.parse_args()
    
    uf = StockUniverseFetcher()
    if args.full:
        tickers = uf.fetch_universe(index_name='NIFTY 500')
    elif args.test:
        tickers = uf.fetch_universe(index_name='NIFTY 50')[:20]
    else:
        tickers = uf.fetch_universe(index_name=args.index or 'NIFTY 200')
        
    run_regime_scan(tickers)

if __name__ == "__main__":
    main()
