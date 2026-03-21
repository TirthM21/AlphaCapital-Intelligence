#!/usr/bin/env python3
"""Run Sector Rotation Analysis for NSE India.

Calculates sectoral index strength relative to Nifty 50 and identifies leader/laggard trends.
"""

import argparse
import logging
import sys
from pathlib import Path
from src.analysis.sector_rotation import SectorRotationAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='NSE Sector Rotation Analysis Tool')
    parser.add_argument('--export', action='store_true', help='Export report to file')
    args = parser.parse_args()
    
    logger.info("Starting Sector Rotation Analysis...")
    
    try:
        analyzer = SectorRotationAnalyzer()
        results = analyzer.analyze_sectors()
        
        report = analyzer.format_report(results)
        
        # Output to console
        print("\n" + report + "\n")
        
        # Export to file if requested
        if args.export:
            output_dir = Path("./data/reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"sector_rotation_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(report)
            
            logger.info(f"Report exported to: {filename}")
            
    except Exception as e:
        logger.error(f"Fatal error during analysis: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
