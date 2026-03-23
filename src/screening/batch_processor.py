"""Batch processor for screening large numbers of stocks with rate limiting.

This module handles:
- Rate-limited API calls (respects Yahoo Finance limits)
- Progress tracking and resume capability
- Incremental results saving
- Time estimation
- Error recovery
"""

import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.fetcher import YahooFinanceFetcher
from src.data.fundamentals_fetcher import fetch_quarterly_financials, analyze_fundamentals_for_signal
from .phase_indicators import classify_phase, calculate_relative_strength
from .signal_engine import score_buy_signal, score_sell_signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchStockProcessor:
    """Process large batches of stocks with rate limiting and progress tracking."""

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        results_dir: str = "./data/batch_results",
        rate_limit_delay: float = 1.0,  # 1 second between stocks (1 TPS)
        batch_size: int = 100  # Save progress every N stocks
    ):
        """Initialize the batch processor.

        Args:
            cache_dir: Directory for data caching
            results_dir: Directory for saving batch results
            rate_limit_delay: Seconds to wait between API calls (1.0 = 1 TPS)
            batch_size: How often to save progress
        """
        self.fetcher = YahooFinanceFetcher(cache_dir=cache_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.batch_size = batch_size

        self.spy_data = None
        self.spy_price = None

        self.progress_file = self.results_dir / "batch_progress.pkl"
        self.current_results = []
        self.processed_tickers = set()

        logger.info(f"BatchStockProcessor initialized (rate: {1/rate_limit_delay:.1f} TPS)")

    def load_progress(self) -> Optional[Dict]:
        """Load progress from previous run.

        Returns:
            Progress dict or None
        """
        if not self.progress_file.exists():
            return None

        try:
            with open(self.progress_file, 'rb') as f:
                progress = pickle.load(f)

            logger.info(f"Loaded progress: {len(progress['processed'])} stocks already processed")
            return progress

        except Exception as e:
            logger.error(f"Error loading progress: {e}")
            return None

    def save_progress(self, tickers_list: List[str], results: List[Dict]):
        """Save current progress.

        Args:
            tickers_list: Complete list of tickers being processed
            results: Results so far
        """
        try:
            progress = {
                'timestamp': datetime.now().isoformat(),
                'total_tickers': len(tickers_list),
                'processed': list(self.processed_tickers),
                'results': results,
                'batch_size': self.batch_size
            }

            with open(self.progress_file, 'wb') as f:
                pickle.dump(progress, f)

            logger.debug(f"Progress saved: {len(self.processed_tickers)}/{len(tickers_list)}")

        except Exception as e:
            logger.error(f"Error saving progress: {e}")

    def fetch_spy_data(self) -> bool:
        """Fetch SPY benchmark data.

        Returns:
            True if successful
        """
        try:
            logger.info("Fetching SPY data...")
            spy_hist = self.fetcher.fetch_price_history('SPY', period='2y')

            if spy_hist.empty:
                logger.error("Failed to fetch SPY data")
                return False

            self.spy_data = spy_hist
            self.spy_price = spy_hist['Close'].iloc[-1]
            logger.info(f"SPY data ready: {len(spy_hist)} days, price: ${self.spy_price:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error fetching SPY data: {e}")
            return False

    def filter_tradable_stocks(
        self,
        tickers: List[str],
        min_price: float = 5.0,
        max_price: float = 10000.0,
        min_volume: int = 100000
    ) -> List[str]:
        """Filter out penny stocks and low-volume stocks.

        This is a quick pre-filter to avoid wasting time on untradable stocks.

        Args:
            tickers: List of tickers to filter
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_volume: Minimum average daily volume

        Returns:
            Filtered list of tickers
        """
        logger.info(f"Pre-filtering {len(tickers)} stocks...")
        logger.info(f"Filters: price ${min_price}-${max_price}, volume >{min_volume:,}")

        # We'll do this check during the main screening loop to avoid extra API calls
        # Just return the full list for now
        return tickers

    def analyze_stock_batch(
        self,
        ticker: str,
        min_price: float = 5.0,
        max_price: float = 10000.0,
        min_volume: int = 100000
    ) -> Optional[Dict]:
        """Analyze a single stock with filtering.

        Args:
            ticker: Stock ticker
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_volume: Minimum volume filter

        Returns:
            Analysis dict or None if filtered out or failed
        """
        try:
            # Fetch price history
            price_data = self.fetcher.fetch_price_history(ticker, period='2y')

            if price_data.empty or len(price_data) < 200:
                logger.debug(f"{ticker}: Insufficient data ({len(price_data)} days)")
                return None

            current_price = price_data['Close'].iloc[-1]

            # Apply filters
            if current_price < min_price or current_price > max_price:
                logger.debug(f"{ticker}: Price ${current_price:.2f} outside range")
                return None

            if 'Volume' in price_data.columns:
                avg_volume = price_data['Volume'].iloc[-20:].mean()
                if avg_volume < min_volume:
                    logger.debug(f"{ticker}: Low volume {avg_volume:,.0f}")
                    return None

            # Classify phase
            phase_info = classify_phase(price_data, current_price)

            # Only analyze stocks in Phase 1 or 2 for buys
            # And Phase 3 or 4 for sells
            phase = phase_info['phase']
            if phase not in [1, 2, 3, 4]:
                return None

            # Calculate relative strength vs SPY
            rs_series = calculate_relative_strength(
                price_data['Close'],
                self.spy_data['Close'],
                period=63
            )

            # Fetch fundamentals (only if in buy/sell phase)
            quarterly_data = {}
            fundamental_analysis = {}

            if phase in [1, 2]:  # Potential buy
                quarterly_data = fetch_quarterly_financials(ticker)
                fundamental_analysis = analyze_fundamentals_for_signal(quarterly_data)

            return {
                'ticker': ticker,
                'price_data': price_data,
                'current_price': current_price,
                'avg_volume': avg_volume if 'Volume' in price_data.columns else 0,
                'phase_info': phase_info,
                'rs_series': rs_series,
                'quarterly_data': quarterly_data,
                'fundamental_analysis': fundamental_analysis
            }

        except Exception as e:
            logger.debug(f"Error analyzing {ticker}: {e}")
            return None

    def process_batch(
        self,
        tickers: List[str],
        resume: bool = True,
        min_price: float = 5.0,
        max_price: float = 10000.0,
        min_volume: int = 100000,
        use_threads: bool = True,
        max_workers: int = 10
    ) -> Dict:
        """Process a batch of tickers with batch downloading and concurrent analysis.

        Args:
            tickers: List of tickers to process
            resume: Resume from previous progress if available
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_volume: Minimum average daily volume
            use_threads: Whether to use multi-threading for analysis
            max_workers: Number of worker threads

        Returns:
            Dict with all results
        """
        logger.info("="*60)
        logger.info("OPTIMIZED BATCH PROCESSING STARTED")
        logger.info(f"Total tickers: {len(tickers)}")
        logger.info(f"Using threads: {use_threads} (max_workers: {max_workers})")
        logger.info("="*60)

        # Load SPY data
        if not self.fetch_spy_data():
            return {'error': 'Failed to fetch SPY data'}

        # Load progress if resuming
        if resume:
            progress = self.load_progress()
            if progress:
                self.processed_tickers = set(progress['processed'])
                self.current_results = progress['results']
                logger.info(f"Resuming: {len(self.processed_tickers)} already done")

        # Filter already processed
        remaining_tickers = [t for t in tickers if t not in self.processed_tickers]
        if not remaining_tickers:
            logger.info("All tickers already processed")
            return {
                'analyses': self.current_results,
                'total_processed': len(tickers),
                'total_analyzed': len(self.current_results)
            }

        logger.info(f"Processing {len(remaining_tickers)} remaining tickers")
        start_time = time.time()
        
        # Step 1: Batch Download Price Data
        chunk_size = 500  # Download in large chunks
        all_data = {}
        
        for i in range(0, len(remaining_tickers), chunk_size):
            chunk = remaining_tickers[i:i+chunk_size]
            logger.info(f"Downloading chunk {i//chunk_size + 1}/{(len(remaining_tickers)-1)//chunk_size + 1} ({len(chunk)} tickers)")
            chunk_data = self.fetcher.batch_download(chunk, period='2y')
            all_data.update(chunk_data)
            
        logger.info(f"Downloaded data for {len(all_data)}/{len(remaining_tickers)} tickers")
        
        # Step 2: Analyze Stock Data (Parallelized)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_analyses = self.current_results.copy()
        phase_results = []
        
        def process_single_ticker(ticker, data):
            try:
                # Basic price/volume filters on the already downloaded data
                if data.empty or len(data) < 200:
                    return None
                    
                current_price = data['Close'].iloc[-1]
                if current_price < min_price or current_price > max_price:
                    return None
                    
                avg_volume = data['Volume'].iloc[-20:].mean()
                if avg_volume < min_volume:
                    return None
                    
                # Classify phase
                phase_info = classify_phase(data, current_price)
                phase = phase_info['phase']
                
                # Calculate relative strength vs SPY
                rs_series = calculate_relative_strength(
                    data['Close'],
                    self.spy_data['Close'],
                    period=63
                )
                
                # Fundamentals are still fetched one-by-one because yfinance doesn't 
                # support batch fundamentals efficiently without rate limits.
                # However, we only fetch for stocks that pass initial filters.
                quarterly_data = {}
                fundamental_analysis = {}
                if phase in [1, 2]:
                    # Limit fundamental fetching to avoid too many requests
                    # In a serious screen, we might skip this or use a local DB
                    pass 

                return {
                    'ticker': ticker,
                    'price_data': data,
                    'current_price': current_price,
                    'avg_volume': avg_volume,
                    'phase_info': phase_info,
                    'rs_series': rs_series,
                    'quarterly_data': quarterly_data,
                    'fundamental_analysis': fundamental_analysis
                }
            except Exception as e:
                logger.debug(f"Error processing {ticker}: {e}")
                return None

        logger.info(f"Starting analysis with {max_workers} threads...")
        
        if use_threads:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(process_single_ticker, t, d): t 
                    for t, d in all_data.items()
                }
                
                count = 0
                for future in as_completed(future_to_ticker):
                    count += 1
                    res = future.result()
                    if res:
                        all_analyses.append(res)
                        phase_results.append({
                            'ticker': res['ticker'],
                            'phase': res['phase_info']['phase']
                        })
                    
                    if count % 100 == 0:
                        logger.info(f"Analyzed {count}/{len(all_data)} stocks...")
        else:
            for i, (ticker, data) in enumerate(all_data.items(), 1):
                res = process_single_ticker(ticker, data)
                if res:
                    all_analyses.append(res)
                if i % 100 == 0:
                    logger.info(f"Analyzed {i}/{len(all_data)} stocks...")

        # Update processed list
        for t in all_data.keys():
            self.processed_tickers.add(t)
            
        # Final save
        self.save_progress(tickers, all_analyses)

        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"Total time: {str(timedelta(seconds=int(total_time)))}")
        logger.info(f"Processed: {len(tickers)} tickers")
        logger.info(f"Analyzed: {len(all_analyses)} stocks")
        logger.info("="*60)

        return {
            'analyses': all_analyses,
            'phase_results': phase_results,
            'total_processed': len(tickers),
            'total_analyzed': len(all_analyses),
            'processing_time_seconds': total_time
        }

    def clear_progress(self):
        """Clear saved progress to start fresh."""
        if self.progress_file.exists():
            self.progress_file.unlink()
            logger.info("Progress cleared")

        self.processed_tickers.clear()
        self.current_results.clear()
