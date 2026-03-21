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
    header = f"{'Ticker':<12} | {'Price':<10} | {'RSI':<6} | {'Signals'}"
    report_lines.append(header)
    report_lines.append("-" * 80)
    
    for res in sorted(results, key=lambda x: len(x.get('signals', [])), reverse=True)[:100]:
        if res.get('signals'):
            sigs_str = ", ".join(res['signals'])
            row = f"{res['ticker']:<12} | {res['price']:<10.2f} | {res['rsi']:<6.1f} | {sigs_str}"
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
    
    # We can use BatchStockProcessor if we want concurrency, but for a standalone script,
    # let's keep it simple or inherit from existing processor logic.
    # Since BatchStockProcessor does a lot of Minervini specific stuff, we'll do a simple loop here
    # or just use yfinance directly for speed in this demonstration.
    
    # Use 1 TPS rate limit for yfinance (roughly)
    import time
    
    for ticker in tickers:
        try:
            # Clean ticker from any irregularities ($, extra whitespace, etc.)
            clean_ticker = ticker.replace("$", "").strip().upper()
            
            # Stage 1: Fetch Data (Check Cache First)
            df = cache.get_price_data(clean_ticker)
            
            if df is None or df.empty:
                logger.info(f"Processing {clean_ticker}...")
                df = pd.DataFrame()
                final_ticker = clean_ticker
                
                # Original fallback logic for downloading
                base_symbol = clean_ticker.split(".")[0]
                suffixes = [".NS", "-SM.NS", ".BO", ""] # Prioritize .NS, then -SM.NS, then .BO, then no suffix
                
                found_data = False
                for suffix in suffixes:
                    current_try = f"{base_symbol}{suffix}" if suffix not in clean_ticker else clean_ticker
                    
                    to_try = [current_try]
                    if suffix == ".NS":
                        to_try.extend([f"{base_symbol}-BE.NS", f"{base_symbol}-BZ.NS"])
                    
                    for t in to_try:
                        logger.info(f"Attempting download for {t}...")
                        for period in ["1y", "1mo", "max"]: # Try 1y, then 1mo, then max
                            df = yf.download(t, period=period, interval="1d", progress=False)
                            if not df.empty:
                                final_ticker = t
                                found_data = True
                                break
                        if found_data: break
                    if found_data: break
                
                if not df.empty:
                    cache.save_price_data(final_ticker, df)
                    ticker = final_ticker # Update ticker to the one that worked
                else:
                    logger.warning(f"No data found for {clean_ticker} after trying all variations")
                    continue
            else:
                logger.info(f"Using cached data for {clean_ticker}")
                ticker = clean_ticker # Ensure ticker is consistent with cache key
            
            # Handle MultiIndex columns if present (from yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Stage 2: Analyze (Check Cache First)
            res = cache.get_signal_result(ticker)
            
            if res is None:
                if args.pipe:
                    res = run_piped_scan(ticker, df, args.pipe)
                else:
                    res = score_extended_signals(ticker, df)
                cache.save_signal_result(ticker, res)
            else:
                logger.info(f"Using cached signal results for {ticker}")
            
            if res.get('signals'):
                # Add date for backtesting
                res['date'] = df.index[-1]
                
                # Perform backtest if requested (on a previous date signal)
                if args.backtest:
                    # Find a signal day from 30 days ago and see how it performed
                    if len(df) > 60:
                        hist_res = score_extended_signals(ticker, df.iloc[:-30])
                        if hist_res.get('signals'):
                            perf = calculate_returns(df, len(df)-31, [1, 5, 10, 22, 30])
                            res.update(perf)
                
                results.append(res)
                
            time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            
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
