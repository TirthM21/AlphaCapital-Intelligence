#!/usr/bin/env python3
"""Execute Advanced Performance Backtest for NSE Strategies.

Validates the Phase 2 / Trend Template strategy against Nifty 50 benchmark
over a 1-year lookback period.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from src.analysis.advanced_backtester import AdvancedBacktester
from src.data.universe_fetcher import StockUniverseFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Advanced Backtesting Engine')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data to backtest')
    parser.add_argument('--symbols', type=int, default=50, help='Number of symbols to include in the simulation pool (0 = full universe)')
    parser.add_argument('--index', type=str, default=None, help='Specific index to backtest (e.g. NIFTY MIDCAP 100)')
    parser.add_argument('--full', action='store_true', help='Use full NSE universe for backtest')
    parser.add_argument('--nifty', action='store_true', help='Use Nifty 50 constituents for backtest')
    parser.add_argument('--fno', action='store_true', help='Use F&O constituents for backtest')
    parser.add_argument('--short', action='store_true', help='Summarize output (<100 lines)')
    parser.add_argument('--cached-only', action='store_true', help='Use cached universe and price data only')
    args = parser.parse_args()
    
    logger.info(f"Preparing advanced backtest for last {args.days} days...")
    
    # 1. Prepare symbols
    uf = StockUniverseFetcher()
    if args.full:
        symbols = uf.fetch_universe(cached_only=args.cached_only)
    elif args.fno:
        symbols = uf.fetch_universe(index_name='SECURITIES IN F&O', cached_only=args.cached_only)
    elif args.index:
        symbols = uf.fetch_universe(index_name=args.index, cached_only=args.cached_only)
    else:
        # Default to Nifty 50 if no flag provided
        symbols = uf.fetch_universe(index_name='NIFTY 50', cached_only=args.cached_only)
        
    if args.symbols > 0:
        symbols = symbols[:args.symbols]
    
    # 2. Setup dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days + 252) # Add extra buffer for SMA 200 calc
    
    # 3. Run simulation
    try:
        engine = AdvancedBacktester()
        results = engine.run_backtest(
            symbols, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'),
            cached_only=args.cached_only,
        )
        
        summary = engine.format_summary(results)
        print("\n" + summary + "\n")
        
        if "daily_values" in results:
            # We could export a CSV of the portfolio curve here
            results['daily_values'].to_csv("./data/reports/backtest_curve_latest.csv")
            logger.info("Portfolio equity curve saved to ./data/reports/backtest_curve_latest.csv")
            
    except Exception as e:
        logger.error(f"Fatal backtest error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
