#!/usr/bin/env python3
"""Run Sector, Market Cap & Thematic Rotation Analysis for NSE India.

Calculates index strength relative to Nifty 50 across:
  - Market Cap indices (Nifty 50/100/200/500, Midcap, Smallcap)
  - Sectoral indices (Bank, IT, Pharma, Auto, Metal, etc.)
  - Thematic/Strategy indices (Alpha 50, Low Vol, Quality, Value, High Beta, ESG)
"""

import argparse
import logging
import sys
from pathlib import Path
from src.analysis.sector_rotation import SectorRotationAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='NSE Rotation Analysis (Market Cap × Sectors × Themes)')
    parser.add_argument('--export', action='store_true', help='Export report to file')
    parser.add_argument('--sectors-only', action='store_true', help='Only analyse sectoral indices')
    args = parser.parse_args()

    logger.info("Starting Rotation Analysis...")

    try:
        analyzer = SectorRotationAnalyzer()

        if args.sectors_only:
            results = analyzer.analyze_sectors()
        else:
            results = analyzer.analyze_all()

        report = analyzer.format_report(results)

        # Output to console
        print("\n" + report + "\n")

        # Export to file if requested
        if args.export:
            output_dir = Path("./data/reports")
            output_dir.mkdir(parents=True, exist_ok=True)

            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"rotation_analysis_{timestamp}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info(f"Report exported to: {filename}")

    except Exception as e:
        logger.error(f"Fatal error during analysis: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
