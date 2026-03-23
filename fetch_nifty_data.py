#!/usr/bin/env python3
"""
Automated Nifty Index Data Fetcher (All Indices)
Uses NiftyIndexFetcher to fetch and save all indices listed in 'index list.json'.
"""

import sys
import os
import time
import logging
import json
from src.data.nifty_indices import NiftyIndexFetcher

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("🚀 Starting Automated Nifty Index Data Fetcher")
    print("=" * 60)

    fetcher = NiftyIndexFetcher(data_dir="./data/indices")
    
    # Get fresh cookies
    if not fetcher.get_fresh_cookies():
        logger.error("Failed to get initial cookies. Exiting.")
        return

    # Load index list
    list_file = "index list.json"
    if not os.path.exists(list_file):
        logger.error(f"Index list file '{list_file}' not found.")
        return
        
    try:
        with open(list_file, 'r', encoding='utf-8-sig') as f:
            index_list_data = json.load(f)
            index_codes = index_list_data.get('d', [])
    except Exception as e:
        logger.error(f"Error loading index list: {e}")
        return

    # Load mapping
    mapping_file = "index mapping.json"
    mapping = {}
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r', encoding='utf-8-sig') as f:
                mapping_data = json.load(f)
                mapping = {item['Index_long_name'].upper(): item['Trading_Index_Name'] for item in mapping_data}
        except Exception as e:
            logger.error(f"Error loading mapping: {e}")

    logger.info(f"Found {len(index_codes)} indices to fetch.")
    print("-" * 60)

    successful = 0
    failed = 0
    
    for i, item in enumerate(index_codes):
        index_name = item.get('indextype', item.get('indexType', ''))
        if not index_name:
            continue
            
        print(f"[{i+1}/{len(index_codes)}] Fetching: {index_name}")
        
        # Determine the trading name to use for the API call
        trading_name = mapping.get(index_name.upper(), index_name)
        if trading_name != index_name:
            logger.debug(f"  Using trading name: {trading_name}")
            
        data = fetcher.fetch_index_data(trading_name)
        
        if data:
            result = fetcher.save_index_data(index_name, data)
            if result.get('success'):
                successful += 1
            else:
                failed += 1
        else:
            logger.error(f"  ✗ Failed to fetch data for {index_name}")
            failed += 1
            
        # Small delay to avoid hammering the server
        time.sleep(1.5)

    print("-" * 60)
    print(f"Basic Summary: {successful} successful, {failed} failed")
    print("✅ Completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user (Ctrl+C). Partial summary above.")
        sys.exit(0)
