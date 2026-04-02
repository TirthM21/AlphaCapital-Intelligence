"""Fetch and maintain the universe of Indian stocks listed on NSE.

This module fetches the complete list of NSE-listed stocks
and maintains a daily-updated universe for screening.
"""

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Optional

import pandas as pd

from .nse_fetcher import NSEFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockUniverseFetcher:
    """Fetches and maintains the universe of NSE-listed stocks."""

    INDEX_SIZE_LIMITS = {
        "NIFTY 50": 75,
        "NIFTY NEXT 50": 75,
        "NIFTY 100": 130,
        "NIFTY 200": 240,
        "NIFTY 500": 550,
        "NIFTY MIDCAP 100": 140,
        "NIFTY SMALLCAP 100": 140,
        "SECURITIES IN F&O": 500,
    }

    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize the universe fetcher.

        Args:
            cache_dir: Directory for caching universe data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "nse_stock_universe.pkl"
        self.nse = NSEFetcher()
        logger.info("StockUniverseFetcher initialized (NSE India)")

    def _get_cache_file(self, include_etfs: bool = False, index_name: Optional[str] = None) -> Path:
        if index_name:
            cache_key = index_name.lower().replace(" ", "_")
        else:
            cache_key = "etf" if include_etfs else "equity"
        return self.cache_dir / f"nse_{cache_key}_universe.pkl"

    def _is_cached_index_size_plausible(self, index_name: Optional[str], symbols: List[str]) -> bool:
        if not index_name:
            return True
        limit = self.INDEX_SIZE_LIMITS.get(index_name.upper())
        return limit is None or len(symbols) <= limit

    def fetch_universe(
        self, 
        force_refresh: bool = False, 
        include_etfs: bool = False,
        index_name: Optional[str] = None,
        cached_only: bool = False,
    ) -> List[str]:
        """Fetch the complete universe of NSE-listed stocks or a specific index.
        
        Args:
            force_refresh: Ignore cache and fetch fresh
            include_etfs: Whether to include ETFs in the universe
            index_name: Specific index to fetch (e.g. 'NIFTY 50', 'NIFTY 100')
            
        Returns:
            List of stock ticker symbols
        """
        # Determine unique cache key
        cache_file = self._get_cache_file(include_etfs=include_etfs, index_name=index_name)
        cache_key = cache_file.stem.replace("nse_", "").replace("_universe", "")

        if cache_file.exists() and (cached_only or not force_refresh):
            cache_age = datetime.now() - datetime.fromtimestamp(
                cache_file.stat().st_mtime
            )

            if cached_only or cache_age < timedelta(days=1):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                cached_symbols = cached_data['symbols']
                if self._is_cached_index_size_plausible(index_name, cached_symbols):
                    logger.info(f"Loaded {len(cached_symbols)} NSE {cache_key} symbols from cache")
                    return cached_symbols
                logger.warning(
                    "Ignoring cached universe for %s because %s symbols is not plausible for that index",
                    index_name,
                    len(cached_symbols),
                )
                if cached_only:
                    return []

        if cached_only:
            logger.warning("Cached-only universe request had no usable cache for %s", index_name or cache_key)
            return []

        if index_name:
            logger.info(f"Fetching index: {index_name}")
            symbols = self.nse.get_index_stocks(index_name)
        elif include_etfs:
            logger.info("Fetching NSE ETF list...")
            symbols = self.nse.get_etfs()
        else:
            logger.info("Fetching total equity market list from NSE CSV...")
            symbols = self.nse.get_all_equity_stocks()

        if not symbols:
            if index_name:
                logger.error("Fetch failed for requested index %s", index_name)
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    cached_symbols = cached_data.get('symbols', [])
                    if self._is_cached_index_size_plausible(index_name, cached_symbols):
                        logger.warning("Using stale cached universe for %s", index_name)
                        return cached_symbols
                return []

            logger.warning(f"Fetch failed for {cache_key}, trying fallback...")
            symbols = self.nse.get_index_stocks('NIFTY 500')

        # Add suffixes for yfinance compatibility
        formatted_symbols = []
        for s in symbols:
            s_clean = str(s).strip().upper()
            
            # Skip invalid symbols or index names returned by some APIs
            if not s_clean or ("NIFTY" in s_clean and " " in s_clean):
                continue
            if s_clean == "NIFTY":
                continue
                
            if "." in s_clean: # Already has a suffix
                formatted_symbols.append(s_clean)
            else:
                formatted_symbols.append(f"{s_clean}.NS")
                
        formatted_symbols = sorted(list(set(formatted_symbols)))

        # Cache the results
        cache_data = {
            'symbols': formatted_symbols,
            'fetch_date': datetime.now().isoformat(),
            'count': len(formatted_symbols)
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"Cached {len(formatted_symbols)} NSE symbols")
        return formatted_symbols

    def get_universe_info(self) -> Dict:
        """Get information about the cached universe."""
        if not self.cache_file.exists():
            return {'cached': False, 'count': 0}

        with open(self.cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        cache_age = datetime.now() - datetime.fromtimestamp(
            self.cache_file.stat().st_mtime
        )

        return {
            'cached': True,
            'count': cached_data['count'],
            'fetch_date': cached_data['fetch_date'],
            'cache_age_hours': cache_age.total_seconds() / 3600
        }
