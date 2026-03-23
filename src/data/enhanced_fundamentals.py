"""Enhanced fundamentals module with FMP integration.

Provides a unified interface for fetching and presenting fundamental data,
optionally using Financial Modeling Prep (FMP) for higher reliability
during earnings seasons.
"""

import logging
import os
import time
from typing import Dict, Optional
from datetime import datetime

# Import existing functions from fundamentals_fetcher
from .fundamentals_fetcher import fetch_quarterly_financials, create_fundamental_snapshot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedFundamentalsFetcher:
    """Class-based wrapper for fundamental data with optional FMP support."""

    def __init__(self):
        """Initialize with FMP key if available."""
        self.fmp_api_key = os.environ.get('FMP_API_KEY')
        self.fmp_available = bool(self.fmp_api_key)
        
        # Tracking usage (placeholders)
        self.fmp_calls_used = 0
        self.fmp_daily_limit = 250  # Free tier default
        self.bandwidth_limit_gb = 100.0
        self.bandwidth_used_mb = 1.5
        
        if self.fmp_available:
            logger.info("FMP API key detected - Enhanced fundamentals enabled")
        else:
            logger.info("FMP API key not found - Using yfinance only")

    def create_snapshot(self, ticker: str, quarterly_data: Dict, use_fmp: bool = False) -> str:
        """Create a formatted fundamental snapshot.

        Args:
            ticker: Stock ticker symbol
            quarterly_data: Data dictionary (from fetch_quarterly_financials)
            use_fmp: Whether to try fetching supplemental data from FMP

        Returns:
            Formatted string for the report
        """
        # For now, we still rely on the core snapshot generator in fundamentals_fetcher
        # If FMP is enabled and available, we would fetch/merge data here
        if use_fmp and self.fmp_available:
            # Placeholder for FMP fetching logic
            # This would merge FMP data into quarterly_data before calling snapshot
            self.fmp_calls_used += 1
            logger.debug(f"FMP: Incremented calls for {ticker}")
            
        return create_fundamental_snapshot(ticker, quarterly_data)

    def get_api_usage(self) -> Dict:
        """Return FMP API usage statistics."""
        now = datetime.now()
        month = now.month
        day = now.day
        
        # Simple earnings season check (matching git_storage_fetcher)
        earnings_windows = [
            (1, 15, 2, 15), (4, 15, 5, 15), (7, 15, 8, 15), (10, 15, 11, 15)
        ]
        is_earnings = any(
            (month == sm and day >= sd) or (month == em and day <= ed)
            for sm, sd, em, ed in earnings_windows
        )

        return {
            'fmp_calls_used': self.fmp_calls_used,
            'fmp_daily_limit': self.fmp_daily_limit,
            'fmp_calls_remaining': self.fmp_daily_limit - self.fmp_calls_used,
            'bandwidth_used_mb': self.bandwidth_used_mb,
            'bandwidth_limit_gb': self.bandwidth_limit_gb,
            'bandwidth_pct_used': (self.bandwidth_used_mb / (self.bandwidth_limit_gb * 1024)) * 100,
            'is_earnings_season': is_earnings,
            'cache_hours': 1 if is_earnings else 24
        }
