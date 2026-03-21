"""
Nifty Factor Index Ranking Tool
================================
Ranks stocks by Alpha, Momentum, Volatility, Beta, Quality, Value,
Dual Momentum, or the NSE Multifactor MQVLv composite.

Usage examples:
    python run_factor_ranking.py --factor alpha --index "NIFTY 500" --symbols 50
    python run_factor_ranking.py --factor dual_momentum --index "NIFTY 200" --symbols 30
    python run_factor_ranking.py --factor multifactor --index "NIFTY 100" --symbols 50
"""
import argparse
import pandas as pd
import yfinance as yf
import logging
import os
from datetime import datetime, timedelta
from src.data.universe_fetcher import StockUniverseFetcher
from src.analysis.factor_indices import NiftyFactorAnalyzer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALL_FACTORS = ['alpha', 'momentum', 'volatility', 'beta',
               'quality', 'value', 'dual_momentum', 'multifactor']

# Maps factor name → sort column and sort direction
FACTOR_CONFIG = {
    'alpha':           {'col': 'alpha',               'asc': False},
    'momentum':        {'col': 'momentum_score',      'asc': False},
    'volatility':      {'col': 'volatility',          'asc': True},   # low vol
    'beta':            {'col': 'beta',                'asc': False},  # high beta
    'quality':         {'col': 'quality_score',       'asc': False},
    'value':           {'col': 'value_score',         'asc': False},
    'dual_momentum':   {'col': 'dual_momentum_score', 'asc': False},
    'multifactor':     {'col': 'composite_score',     'asc': False},
}

DISPLAY_COLS = {
    'alpha':           ['symbol', 'alpha', 'alpha_score', 'volatility'],
    'momentum':        ['symbol', 'momentum_score', 'mr12', 'mr6', 'volatility'],
    'volatility':      ['symbol', 'volatility', 'low_vol_score'],
    'beta':            ['symbol', 'beta', 'alpha', 'volatility'],
    'quality':         ['symbol', 'quality_score', 'roe', 'de', 'sector'],
    'value':           ['symbol', 'value_score', 'ep', 'bp', 'div_yield'],
    'dual_momentum':   ['symbol', 'dual_momentum_score', 'momentum_score',
                        'abs_mom_12m', 'mr12'],
    'multifactor':     ['symbol', 'composite_score', 'aggregate_percentile',
                        'momentum_score', 'quality_score', 'value_score',
                        'low_vol_score'],
}


def main():
    parser = argparse.ArgumentParser(
        description='Nifty Factor Index Ranking Tool')
    parser.add_argument('--index', type=str, default='NIFTY 500',
                        help='Index universe (e.g. NIFTY 100, NIFTY 200, NIFTY 500)')
    parser.add_argument('--factor', type=str, default='alpha',
                        choices=ALL_FACTORS,
                        help='Factor to rank by')
    parser.add_argument('--symbols', type=int, default=50,
                        help='Top N stocks to return')
    parser.add_argument('--days', type=int, default=400,
                        help='Historical data window (≥366 for 1-year factors)')
    args = parser.parse_args()

    logger.info(f"Factor: {args.factor.upper()}  |  Universe: {args.index}  |  Top {args.symbols}")

    # ── 1. Fetch symbol universe ──────────────────────────────────────
    uf = StockUniverseFetcher()
    symbols = uf.fetch_universe(index_name=args.index)
    if not symbols:
        logger.error(f"No symbols found for {args.index}"); return

    # ── 2. Fetch benchmark (Nifty 50) ────────────────────────────────
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=args.days + 30)
    logger.info("Downloading Nifty 50 benchmark...")
    bench = yf.download('^NSEI', start=start_dt, end=end_dt)
    if bench.empty:
        logger.error("Benchmark download failed!"); return

    # ── 3. Fetch stock data in batches ────────────────────────────────
    batch_sz = 50
    all_data = {}
    for i in range(0, len(symbols), batch_sz):
        batch = symbols[i:i + batch_sz]
        tickers = [s if s.endswith('.NS') else f"{s}.NS" for s in batch]
        try:
            raw = yf.download(tickers, start=start_dt, end=end_dt,
                              group_by='ticker', threads=True)
            for s in batch:
                t = s if s.endswith('.NS') else f"{s}.NS"
                if t in raw:
                    sdf = raw[t].dropna()
                    if not sdf.empty:
                        all_data[s] = sdf
            logger.info(f"Batch {i // batch_sz + 1}/"
                        f"{len(symbols) // batch_sz + 1} - "
                        f"{len(all_data)} stocks loaded")
        except Exception as e:
            logger.error(f"Batch error: {e}")

    if not all_data:
        logger.warning("No stock data downloaded."); return

    # ── 4. Compute factor scores ──────────────────────────────────────
    analyzer = NiftyFactorAnalyzer()
    # Pass the requested factor so the engine knows which extras to compute
    results = analyzer.compute_scores(all_data, bench, factors=[args.factor])

    if results.empty:
        logger.warning("No results after factor computation."); return

    # ── 5. Rank, filter, save ─────────────────────────────────────────
    cfg = FACTOR_CONFIG[args.factor]
    if cfg['col'] not in results.columns:
        logger.error(f"Column {cfg['col']} not found. Available: {list(results.columns)}")
        return

    ranked = results.sort_values(by=cfg['col'], ascending=cfg['asc'])
    top_n = ranked.head(args.symbols)

    # Display
    disp_cols = [c for c in DISPLAY_COLS.get(args.factor, ['symbol']) if c in top_n.columns]
    header = f"TOP {args.symbols} STOCKS BY {args.factor.upper()} ({args.index})"
    logger.info(f"\n{'=' * 80}\n{header}\n{'=' * 80}")
    print(top_n[disp_cols].to_string(index=False))

    # Save
    out_dir = "data/reports/rankings"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{args.factor}_ranking_{args.index.replace(' ', '_')}_{datetime.now():%Y%m%d}.csv"
    path = os.path.join(out_dir, fname)
    top_n.to_csv(path, index=False)
    logger.info(f"\nFull report saved → {path}")


if __name__ == "__main__":
    main()
