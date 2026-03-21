"""Sector Rotation Analysis for NSE India.

Analyzes sectoral index strength relative to Nifty 50 to identify leader and laggard sectors.
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Sectoral Index mapping (Yahoo Finance tickers)
NSE_SECTORS = {
    "NIFTY 50 (BENCHMARK)": "^NSEI",
    "BANKING": "^NSEBANK",
    "IT": "^CNXIT",
    "PHARMA": "^CNXPHARMA",
    "AUTO": "^CNXAUTO",
    "FMCG": "^CNXFMCG",
    "METAL": "^CNXMETAL",
    "REALTY": "^CNXREALTY",
    "ENERGY": "^CNXENERGY",
    "INFRASTRUCTURE": "^CNXINFRA",
    "PSU BANK": "^CNXPSUBANK",
    "FINANCIAL SERVICES": "^CNXFIN",
    "MEDIA": "^CNXMEDIA",
    "COMMODITIES": "^CNXCOMM",
    "SERVICES": "^CNXSERVICE",
    "CONSUMER DURABLES": "^CNXCONSUMER"
}

class SectorRotationAnalyzer:
    """Analyze sector performance and rotation."""

    def __init__(self, lookback_days: int = 126): # ~6 months
        self.lookback_days = lookback_days
        self.benchmark = "^NSEI"

    def fetch_index_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch index prices from yfinance."""
        try:
            df = yf.download(ticker, period=period, interval="1d", progress=False)
            if df.empty: return pd.DataFrame()
            # Handle MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def calculate_relative_strength(self, sector_price: pd.Series, bench_price: pd.Series) -> pd.Series:
        """Calculate RS line (Sector/Benchmark)."""
        # Align dates
        combined = pd.concat([sector_price, bench_price], axis=1).dropna()
        combined.columns = ['sector', 'bench']
        return (combined['sector'] / combined['bench']) * 100

    def analyze_sectors(self) -> Dict[str, any]:
        """Perform full sector strength analysis."""
        logger.info("Fetching sectoral index data...")
        
        # Bench data
        bench_df = self.fetch_index_data(self.benchmark)
        if bench_df.empty: return {"error": "Failed to fetch benchmark data"}
        bench_close = bench_df['Close']

        results = []
        
        for sector_name, ticker in NSE_SECTORS.items():
            if ticker == self.benchmark: continue
            
            logger.info(f"Analyzing {sector_name} ({ticker})...")
            sector_df = self.fetch_index_data(ticker)
            if sector_df.empty: continue
            
            close = sector_df['Close']
            
            # 1. Performance over different lookbacks
            perf_1m = ((close.iloc[-1] / close.iloc[-22]) - 1) * 100 if len(close) > 22 else 0
            perf_3m = ((close.iloc[-1] / (close.iloc[-63] if len(close) > 63 else close.iloc[0])) - 1) * 100
            
            # 2. RS Line Analysis
            rs_line = self.calculate_relative_strength(close, bench_close)
            rs_slope_1mo = (rs_line.iloc[-1] / rs_line.iloc[-22] - 1) * 100 if len(rs_line) > 22 else 0
            
            # 3. Distance from 50 SMA (Trend Momentum)
            sma_50 = close.rolling(50).mean()
            dist_sma50 = ((close.iloc[-1] / sma_50.iloc[-1]) - 1) * 100 if not pd.isna(sma_50.iloc[-1]) else 0
            
            # 4. Phase check (Simple logic: Price vs 50 EMA)
            ema_20 = close.ewm(span=20).mean().iloc[-1]
            status = "Strong" if close.iloc[-1] > ema_20 and rs_slope_1mo > 0 else ("Weak" if close.iloc[-1] < ema_20 else "Neutral")

            results.append({
                "sector": sector_name,
                "ticker": ticker,
                "perf_1m": round(perf_1m, 2),
                "perf_3m": round(perf_3m, 2),
                "rs_slope": round(rs_slope_1mo, 2),
                "dist_sma50": round(dist_sma50, 2),
                "status": status,
                "score": round((perf_1m + rs_slope_1mo * 2 + dist_sma50), 2) # Weighted score
            })

        # Sort by score (strength)
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return {"timestamp": datetime.now().isoformat(), "sectors": results}

    def format_report(self, analysis: Dict) -> str:
        """Format the analysis into a readable report."""
        if "error" in analysis: return f"Error: {analysis['error']}"
        
        report = []
        report.append("="*80)
        report.append("NSE SECTOR ROTATION ANALYSIS")
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        report.append("")
        
        header = f"{'Sector':<22} | {'Status':<10} | {'Perf 1M':<10} | {'RS Slope':<10} | {'Strength Score'}"
        report.append(header)
        report.append("-" * 80)
        
        sectors = analysis['sectors']
        
        for s in sectors:
            status_emoji = "🔥" if s['status'] == "Strong" else ("❄️" if s['status'] == "Weak" else "➖")
            line = f"{s['sector']:<22} | {status_emoji} {s['status']:<7} | {s['perf_1m']:<10.2f} | {s['rs_slope']:<10.2f} | {s['score']}"
            report.append(line)
            
        report.append("\n" + "="*80)
        report.append("ROTATION INSIGHTS:")
        
        leading = [s['sector'] for s in sectors[:3]]
        lagging = [s['sector'] for s in sectors[-3:]]
        
        report.append(f"🔝 Leading Sectors (Institutional Flow): {', '.join(leading)}")
        report.append(f"🔻 Lagging Sectors (Avoid for now): {', '.join(lagging)}")
        
        # Potential Rotation Candidates
        rotation_candidates = [s['sector'] for s in sectors if s['rs_slope'] > 0 and s['perf_1m'] < 0]
        if rotation_candidates:
            report.append(f"🔄 Potential Rotation/Bottoming Candidates: {', '.join(rotation_candidates[:3])}")
            
        return "\n".join(report)
