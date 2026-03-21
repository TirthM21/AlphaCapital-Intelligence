"""Persistent market data cache for Alpha Capital Intelligence.

Stores price history and technical analysis results to speed up repeated scans.
"""

import os
import json
import pickle
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

class MarketDataCache:
    """Cache for price history and technical analysis results."""
    
    def __init__(self, cache_dir: str = "./data/market_cache"):
        self.cache_dir = Path(cache_dir)
        self.price_dir = self.cache_dir / "prices"
        self.signals_dir = self.cache_dir / "signals"
        
        # Ensure directories exist
        self.price_dir.mkdir(parents=True, exist_ok=True)
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MarketDataCache initialized at {cache_dir}")

    def get_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Retrieve cached price data if it matches today's date."""
        cache_file = self.price_dir / f"{ticker.replace('.', '_')}.pkl"
        
        if cache_file.exists():
            try:
                # Check modification time to see if it was updated today
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime).date()
                if mtime == date.today():
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to read price cache for {ticker}: {e}")
        return None

    def save_price_data(self, ticker: str, df: pd.DataFrame):
        """Save price data to cache."""
        if df.empty: return
        cache_file = self.price_dir / f"{ticker.replace('.', '_')}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            logger.error(f"Failed to save price cache for {ticker}: {e}")

    def get_signal_result(self, ticker: str, strategy: str = "extended") -> Optional[Dict]:
        """Retrieve cached signal result from today."""
        cache_file = self.signals_dir / f"{ticker.replace('.', '_')}_{strategy}.json"
        
        if cache_file.exists():
            try:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime).date()
                if mtime == date.today():
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            except Exception as e:
                pass
        return None

    def save_signal_result(self, ticker: str, result: Dict, strategy: str = "extended"):
        """Save signal result to cache."""
        cache_file = self.signals_dir / f"{ticker.replace('.', '_')}_{strategy}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save signal cache for {ticker}: {e}")

    def clear_old_cache(self, days: int = 1):
        """Clear cache older than X days."""
        # Implementation for cleaning up old files to save disk space
        pass
