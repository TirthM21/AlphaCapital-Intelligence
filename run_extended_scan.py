#!/usr/bin/env python3
"""Extended Market Scanner for NSE and Diverse signals.

Supports multiple indices, IPOs, and extended technical signals.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import yfinance as yf

from src.data.universe_fetcher import StockUniverseFetcher
from src.screening.extended_signals import score_extended_signals
from src.screening.piped_scanners import run_piped_scan
from src.screening.batch_processor import BatchStockProcessor
from src.notifications.telegram_notifier import TelegramNotifier
from src.analysis.backtest_engine import calculate_returns, format_performance_summary
from src.data.market_cache import MarketDataCache
from src.data.fetcher import YahooFinanceFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_report(results: list, title: str):
    """Generate and display a report of the signals found."""
    # Ensure reports directory exists
    report_dir = Path("./data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = report_dir / f"scan_{title.replace(' ', '_')}_{timestamp}.txt"
    json_filename = report_dir / f"scan_{title.replace(' ', '_')}_{timestamp}.json"
    
    report_lines = []
    report_lines.append("\n" + "="*80)
    report_lines.append(f"EXTENDED SCAN REPORT: {title}")
    report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    
    if not results:
        report_lines.append("No signals found.")
        print("\n".join(report_lines))
        return

    # Count signals
    all_signals = {}
    for res in results:
        for sig in res.get('signals', []):
            all_signals[sig] = all_signals.get(sig, 0) + 1
            
    report_lines.append("\nSIGNAL SUMMARY:")
    for sig, count in sorted(all_signals.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"  • {sig}: {count} stocks")
    
    report_lines.append("\nDETAILED SIGNALS (Top 100):")
    for res in sorted(results, key=lambda x: len(x.get('signals', [])), reverse=True)[:100]:
        if res.get('signals'):
            price = res.get('price', 0)
            # Suggested levels if not already in result
            stop = res.get('stop', price * 0.95)
            target = res.get('target', price * 1.10)
            
            sigs_str = ", ".join(res['signals'][:3]) # Top 3 signals
            row = f"{res['ticker']:<12} | {price:<10.2f} | {res['rsi']:<6.1f} | S: {stop:<8.2f} T: {target:<8.2f} | {sigs_str}"
            report_lines.append(row)
            
    # Output to console
    print("\n".join(report_lines))
    
    # Save to file
    with open(filename, 'w') as f:
        f.write("\n".join(report_lines))
    
    # Save to JSON
    import json
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    logger.info(f"Report saved to {filename}")
    logger.info(f"Full data saved to {json_filename}")

def main():
    parser = argparse.ArgumentParser(description='Extended NSE Market Scanner')
    parser.add_argument('--index', type=str, default=None, 
                       help='Index to scan (e.g. NIFTY 500, NIFTY 50, NIFTY MIDCAP 100, etc.)')
    parser.add_argument('--full', action='store_true', help='Scan ALL NSE stocks')
    parser.add_argument('--ipo', action='store_true', help='Scan newly listed stocks (past 2 years)')
    parser.add_argument('--backtest', action='store_true', help='Perform historic backtest on results')
    parser.add_argument('--fno', action='store_true', help='Scan F&O stocks')
    parser.add_argument('--nasdaq', action='store_true', help='Scan NASDAQ 100 stocks')
    parser.add_argument('--midcap', action='store_true', help='Scan Nifty Midcap 100')
    parser.add_argument('--smallcap', action='store_true', help='Scan Nifty Smallcap 100')
    parser.add_argument('--nifty', action='store_true', help='Scan Nifty 50')
    parser.add_argument('--test', action='store_true', help='Test mode (limit to 10 stocks)')
    parser.add_argument('--pipe', type=int, choices=[1, 2, 3, 4], help='Run piped scanner (1-4)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Extended Scan...")
    
    universe_fetcher = StockUniverseFetcher()
    
    if args.full:
        tickers = universe_fetcher.fetch_universe()
        title = "ALL NSE EQUITIES"
    elif args.nasdaq:
        # For NASDAQ, we'll fetch some top 50 symbols for demo
        tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "NFLX", "IBKR", "AMD"]
        title = "NASDAQ 100"
    elif args.midcap:
        tickers = universe_fetcher.fetch_universe(index_name='NIFTY MIDCAP 100')
        title = "NIFTY MIDCAP 100"
    elif args.smallcap:
        tickers = universe_fetcher.fetch_universe(index_name='NIFTY SMALLCAP 100')
        title = "NIFTY SMALLCAP 100"
    elif args.nifty:
        tickers = universe_fetcher.fetch_universe(index_name='NIFTY 50')
        title = "NIFTY 50"
    elif args.ipo:
        tickers = universe_fetcher.nse.get_ipos(type='past')
        title = "RECENT IPOs (PAST 2 YEARS)"
    elif args.fno:
        tickers = universe_fetcher.fetch_universe(index_name='SECURITIES IN F&O')
        title = "F&O STOCKS"
    elif args.index:
        tickers = universe_fetcher.fetch_universe(index_name=args.index)
        title = args.index
    else:
        # Default to full market if no specific flag OR index provided
        tickers = universe_fetcher.fetch_universe()
        title = "ALL NSE EQUITIES (FULL MARKET)"
        
    if args.test:
        tickers = tickers[:10]
        title += " (TEST)"
        
    logger.info(f"Scanning {len(tickers)} stocks...")
    
    # Ensure all tickers have a suffix (.NS default)
    tickers = [f"{t}.NS" if "." not in t else t for t in tickers]
    
    results = []
    cache = MarketDataCache()
    fetcher = YahooFinanceFetcher()
    
    # --- OPTIMIZED BATCH PROCESSING ---
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # 1. Check for cached results first to avoid unnecessary downloads
    remaining_tickers = []
    for ticker in tickers:
        res = cache.get_signal_result(ticker)
        if res:
            results.append(res)
        else:
            remaining_tickers.append(ticker)
            
    if not remaining_tickers:
        logger.info("All stocks retrieved from cache.")
    else:
        logger.info(f"Processing {len(remaining_tickers)} stocks (using batch download and threads)...")
        
        # 2. Batch Download in chunks
        chunk_size = 300
        all_data = {}
        for i in range(0, len(remaining_tickers), chunk_size):
            chunk = remaining_tickers[i:i+chunk_size]
            logger.info(f"Downloading chunk {i//chunk_size + 1}/{(len(remaining_tickers)-1)//chunk_size + 1}...")
            chunk_data = fetcher.batch_download(chunk, period="1y")
            all_data.update(chunk_data)
            
        # 3. Parallel Analysis
        def analyze_ticker(ticker, df):
            try:
                if df is None or df.empty:
                    return None
                    
                # Standardize columns if MultiIndex (though batch_download should have handled it)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if args.pipe:
                    res = run_piped_scan(ticker, df, args.pipe)
                else:
                    res = score_extended_signals(ticker, df)
                
                if res.get('signals'):
                    res['date'] = df.index[-1]
                    # Backtest logic if requested
                    if args.backtest and len(df) > 60:
                        hist_res = score_extended_signals(ticker, df.iloc[:-30])
                        if hist_res.get('signals'):
                            perf = calculate_returns(df, len(df)-31, [1, 5, 10, 22, 30])
                            res.update(perf)
                    
                    cache.save_signal_result(ticker, res)
                    # We also want to cache the price data if we just downloaded it
                    cache.save_price_data(ticker, df)
                    return res
                return None
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                return None

        max_workers = 15
        logger.info(f"Analyzing {len(all_data)} stocks with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(analyze_ticker, t, d): t 
                for t, d in all_data.items()
            }
            
            for future in as_completed(future_to_ticker):
                res = future.result()
                if res:
                    results.append(res)
    # --- END OPTIMIZED BATCH PROCESSING ---
            
    generate_report(results, title)
    
    if args.backtest and results:
        print("\n" + "="*80)
        print("HISTORIC BACKTEST RESULTS (Signals from 30 days ago)")
        print("="*80)
        print(format_performance_summary(results))

    # Optional Telegram Notification
    telegram = TelegramNotifier()
    if results:
        telegram.send_signals(results, title)

if __name__ == '__main__':
    main()
