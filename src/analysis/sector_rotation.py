"""Sector, Market Cap & Thematic Rotation Analysis for NSE India.

Analyzes sectoral, market-cap, and strategy index strength relative to Nifty 50
to identify leader and laggard rotations across all dimensions.
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── CORE NIFTY MARKET CAP INDICES ──────────────────────────────────────────
MARKET_CAP_INDICES = {
    "Nifty 50": "^NSEI",
    "Nifty Next 50": "^NSMIDCP",
    "Nifty 100": "^CNX100",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
    "Nifty Midcap 50": "^NSEMDCP50",
    "Nifty Midcap 100": "^CNXMCAP",
    "Nifty Smallcap 100": "^CNXSC",
}

# ─── SECTORAL INDICES ──────────────────────────────────────────────────────
NSE_SECTORS = {
    "Bank Nifty": "^NSEBANK",
    "Financial Services": "^CNXFIN",
    "Private Bank": "^NSEBANKPVT",
    "PSU Bank": "^CNXPSUBANK",
    "IT": "^CNXIT",
    "Pharma": "^CNXPHARMA",
    "FMCG": "^CNXFMCG",
    "Auto": "^CNXAUTO",
    "Metal": "^CNXMETAL",
    "Realty": "^CNXREALTY",
    "Media": "^CNXMEDIA",
    "Oil & Gas / Energy": "^CNXENERGY",
    "Healthcare": "^CNXHEALTHCARE",
    "Consumption": "^CNXCONSUMPTION",
}

# ─── THEMATIC / STRATEGY INDICES ───────────────────────────────────────────
THEMATIC_INDICES = {
    "Alpha 50": "^CNXALPHA",
    "Low Volatility 50": "^CNXLOWVOL",
    "Quality 30": "^CNXQUALITY",
    "Value 20": "^CNXVALUE",
    "Growth Sectors 15": "^CNXGROWTH",
    "Dividend Opp 50": "^CNXDIVOPP",
    "Alpha Low Vol 30": "^CNXALPHALOWVOL",
    "High Beta 50": "^CNXBETA",
    "100 ESG": "^CNXESG100",
}

# Combine all for full rotation analysis
ALL_INDEX_GROUPS = {
    "MARKET CAP": MARKET_CAP_INDICES,
    "SECTORAL": NSE_SECTORS,
    "THEMATIC / STRATEGY": THEMATIC_INDICES,
}


class SectorRotationAnalyzer:
    """Analyze sector, market-cap, and thematic index performance & rotation."""

    def __init__(self, lookback_days: int = 126):  # ~6 months
        self.lookback_days = lookback_days
        self.benchmark = "^NSEI"

    def fetch_index_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch index prices from yfinance."""
        try:
            df = yf.download(ticker, period=period, interval="1d", progress=False)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def calculate_relative_strength(self, sector_price: pd.Series, bench_price: pd.Series) -> pd.Series:
        """Calculate RS line (Sector/Benchmark)."""
        combined = pd.concat([sector_price, bench_price], axis=1).dropna()
        combined.columns = ['sector', 'bench']
        return (combined['sector'] / combined['bench']) * 100

    def _analyze_group(self, index_map: Dict[str, str], bench_close: pd.Series) -> List[Dict]:
        """Analyze a group of indices against the benchmark."""
        results = []

        for name, ticker in index_map.items():
            if ticker == self.benchmark:
                continue

            logger.info(f"  Analyzing {name} ({ticker})...")
            df = self.fetch_index_data(ticker)
            if df.empty or len(df) < 23:
                logger.warning(f"  Skipping {name} — insufficient data")
                continue

            close = df['Close']

            # 1. Performance over different lookbacks
            perf_1w = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100 if len(close) > 5 else 0
            perf_1m = ((close.iloc[-1] / close.iloc[-22]) - 1) * 100 if len(close) > 22 else 0
            perf_3m = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100 if len(close) > 63 else 0
            perf_6m = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100 if len(close) > 126 else 0

            # 2. RS Line Analysis
            rs_line = self.calculate_relative_strength(close, bench_close)
            rs_slope_1mo = (rs_line.iloc[-1] / rs_line.iloc[-22] - 1) * 100 if len(rs_line) > 22 else 0

            # 3. Distance from 50 SMA (Trend Momentum)
            sma_50 = close.rolling(50).mean()
            dist_sma50 = ((close.iloc[-1] / sma_50.iloc[-1]) - 1) * 100 if not pd.isna(sma_50.iloc[-1]) else 0

            # 4. Phase check (Simple: Price vs 20 EMA + RS slope direction)
            ema_20 = close.ewm(span=20).mean().iloc[-1]
            if close.iloc[-1] > ema_20 and rs_slope_1mo > 0:
                status = "Strong"
            elif close.iloc[-1] < ema_20 and rs_slope_1mo < 0:
                status = "Weak"
            else:
                status = "Neutral"

            results.append({
                "name": name,
                "ticker": ticker,
                "perf_1w": round(perf_1w, 2),
                "perf_1m": round(perf_1m, 2),
                "perf_3m": round(perf_3m, 2),
                "perf_6m": round(perf_6m, 2),
                "rs_slope": round(rs_slope_1mo, 2),
                "dist_sma50": round(dist_sma50, 2),
                "status": status,
                "score": round((perf_1m + rs_slope_1mo * 2 + dist_sma50), 2),
            })

        return sorted(results, key=lambda x: x['score'], reverse=True)

    def analyze_all(self) -> Dict[str, any]:
        """Perform full rotation analysis across market cap, sectors, and themes."""
        logger.info("Fetching Nifty 50 benchmark data...")
        bench_df = self.fetch_index_data(self.benchmark)
        if bench_df.empty:
            return {"error": "Failed to fetch benchmark data"}
        bench_close = bench_df['Close']

        all_results = {}
        for group_name, index_map in ALL_INDEX_GROUPS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyzing {group_name} indices...")
            logger.info(f"{'='*60}")
            all_results[group_name] = self._analyze_group(index_map, bench_close)

        return {
            "timestamp": datetime.now().isoformat(),
            "groups": all_results,
        }

    # Keep backward-compatible method
    def analyze_sectors(self) -> Dict[str, any]:
        """Backward compatible: analyze only sectors."""
        full = self.analyze_all()
        if "error" in full:
            return full
        # Flatten all groups for backward compat
        all_items = []
        for group_results in full["groups"].values():
            all_items.extend(group_results)
        return {"timestamp": full["timestamp"], "sectors": all_items}

    def format_report(self, analysis: Dict) -> str:
        """Format the analysis into a readable report."""
        if "error" in analysis:
            return f"Error: {analysis['error']}"

        # If called with analyze_all() result
        if "groups" in analysis:
            return self._format_full_report(analysis)

        # Legacy format (analyze_sectors)
        return self._format_legacy_report(analysis)

    def _format_full_report(self, analysis: Dict) -> str:
        """Format the full multi-group rotation report."""
        report = []
        report.append("=" * 90)
        report.append("NSE ROTATION ANALYSIS — Market Cap × Sectors × Thematic Strategies")
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 90)

        for group_name, items in analysis["groups"].items():
            report.append("")
            report.append(f"{'━' * 90}")
            report.append(f"  {group_name} ROTATION")
            report.append(f"{'━' * 90}")

            header = f"  {'Index':<25} {'Status':<10} {'1W':>7} {'1M':>7} {'3M':>7} {'6M':>7} {'RS Slope':>9} {'Score':>8}"
            report.append(header)
            report.append("  " + "-" * 86)

            for s in items:
                emoji = "🔥" if s['status'] == "Strong" else ("❄️" if s['status'] == "Weak" else "➖")
                line = (
                    f"  {s['name']:<25} {emoji} {s['status']:<7}"
                    f" {s['perf_1w']:>6.1f}%"
                    f" {s['perf_1m']:>6.1f}%"
                    f" {s['perf_3m']:>6.1f}%"
                    f" {s['perf_6m']:>6.1f}%"
                    f" {s['rs_slope']:>8.2f}"
                    f" {s['score']:>8.1f}"
                )
                report.append(line)

            # Group-level insights
            if items:
                leading = [s['name'] for s in items[:3]]
                lagging = [s['name'] for s in items[-3:]]
                rotation = [s['name'] for s in items if s['rs_slope'] > 0 and s['perf_1m'] < 0]

                report.append("")
                report.append(f"  🔝 Leading: {', '.join(leading)}")
                report.append(f"  🔻 Lagging: {', '.join(lagging)}")
                if rotation:
                    report.append(f"  🔄 Bottoming/Rotation: {', '.join(rotation[:3])}")

        report.append("")
        report.append("=" * 90)
        report.append("ROTATION MATRIX LEGEND:")
        report.append("  🔥 Strong = Price > 20 EMA AND RS Slope > 0 (outperforming Nifty 50)")
        report.append("  ❄️ Weak   = Price < 20 EMA AND RS Slope < 0 (underperforming Nifty 50)")
        report.append("  ➖ Neutral = Mixed signals")
        report.append("  Score     = 1M Performance + 2 × RS Slope + Distance from 50 SMA")
        report.append("=" * 90)

        return "\n".join(report)

    def _format_legacy_report(self, analysis: Dict) -> str:
        """Legacy format for backward compatibility."""
        if "error" in analysis:
            return f"Error: {analysis['error']}"

        report = []
        report.append("=" * 80)
        report.append("NSE SECTOR ROTATION ANALYSIS")
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")

        header = f"{'Index':<22} | {'Status':<10} | {'Perf 1M':<10} | {'RS Slope':<10} | {'Strength Score'}"
        report.append(header)
        report.append("-" * 80)

        sectors = analysis['sectors']

        for s in sectors:
            emoji = "🔥" if s['status'] == "Strong" else ("❄️" if s['status'] == "Weak" else "➖")
            line = f"{s['name']:<22} | {emoji} {s['status']:<7} | {s['perf_1m']:<10.2f} | {s['rs_slope']:<10.2f} | {s['score']}"
            report.append(line)

        report.append("\n" + "=" * 80)
        report.append("ROTATION INSIGHTS:")

        leading = [s['name'] for s in sectors[:3]]
        lagging = [s['name'] for s in sectors[-3:]]

        report.append(f"🔝 Leading (Institutional Flow): {', '.join(leading)}")
        report.append(f"🔻 Lagging (Avoid for now): {', '.join(lagging)}")

        rotation_candidates = [s['name'] for s in sectors if s['rs_slope'] > 0 and s['perf_1m'] < 0]
        if rotation_candidates:
            report.append(f"🔄 Potential Rotation/Bottoming: {', '.join(rotation_candidates[:3])}")

        return "\n".join(report)
