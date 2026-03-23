#!/usr/bin/env python3
"""
Single Nifty Index Fetcher
Fetch and save data for a specific Nifty index from the command line.
Example: python fetch_single_nifty.py 'NIFTY 50'
"""

import sys
import os
import json
import logging
from src.data.nifty_indices import NiftyIndexFetcher

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_single_nifty.py <INDEX_NAME>")
        print("Example: python fetch_single_nifty.py 'NIFTY 50'")
        print("Example: python fetch_single_nifty.py 'NIFTY BANK'")
        return
    
    index_name = sys.argv[1]
    
    print(f"🚀 Fetching data for: {index_name}")
    print("=" * 60)
    
    fetcher = NiftyIndexFetcher(data_dir="./data/indices")
    
    # Get fresh cookies
    if not fetcher.get_fresh_cookies():
        logger.error("Failed to get fresh cookies. Exiting.")
        return
    
    # Load mapping to check if a trading name exists
    mapping_file = "index mapping.json"
    trading_name = index_name
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r', encoding='utf-8-sig') as f:
                mapping_data = json.load(f)
                mapping = {item['Index_long_name'].upper(): item['Trading_Index_Name'] for item in mapping_data}
                trading_name = mapping.get(index_name.upper(), index_name)
        except Exception as e:
            logger.error(f"Error loading mapping: {e}")
    
    if trading_name != index_name:
        logger.info(f"Using trading name: {trading_name}")
    
    # Fetch the data
    data = fetcher.fetch_index_data(trading_name)
    
    if data:
        result = fetcher.save_index_data(index_name, data)
        if result.get('success'):
            print(f"✅ Successfully fetched and saved data for '{index_name}'")
            print(f"   Saved to: {result['filepath']}")
        else:
            print(f"❌ Failed to save data for '{index_name}'")
    else:
        print(f"❌ Failed to fetch data for '{index_name}'")

if __name__ == "__main__":
    main()
