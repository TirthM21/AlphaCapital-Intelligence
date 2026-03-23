#!/usr/bin/env python3
"""
Automated Nifty Index Data Fetcher
Uses curl subprocess for reliable TLS handling with Akamai CDN.
Specifically designed for NSE India indices from niftyindices.com.
"""

import json
import subprocess
import time
import os
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

class NiftyIndexFetcher:
    """Fetches index data from niftyindices.com using curl for better compatibility."""
    
    def __init__(self, data_dir: str = "./data/indices"):
        self.base_url = "https://www.niftyindices.com"
        self.api_url = f"{self.base_url}/Backpage.aspx/getTotalReturnIndexString"
        # Use system temp directory for cookies to support Windows/Linux
        self.cookie_file = os.path.join(tempfile.gettempdir(), "nifty_cookies.txt")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"NiftyIndexFetcher initialized. Data dir: {self.data_dir}")

    def _curl(self, url: str, method: str = "GET", data: Any = None, timeout: int = 30):
        """Execute a curl request and return (status_code, body)"""
        # Ensure curl is available
        cmd = [
            "curl", "-s",
            "--connect-timeout", "10",
            "--max-time", str(timeout),
            "-b", self.cookie_file,
            "-c", self.cookie_file,
            "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "-H", "Accept-Language: en-GB,en-US;q=0.9,en;q=0.8",
            "-w", "\n__HTTP_STATUS__%{http_code}",
        ]

        if method == "POST" and data is not None:
            cmd += [
                "-X", "POST",
                "-H", "Accept: application/json, text/javascript, */*; q=0.01",
                "-H", "Content-Type: application/json; charset=UTF-8",
                "-H", "X-Requested-With: XMLHttpRequest",
                "-H", f"Origin: {self.base_url}",
                "-H", f"Referer: {self.base_url}/reports/historical-data",
                "-d", json.dumps(data),
            ]
        else:
            cmd += ["-H", "Accept: text/html,application/xhtml+xml"]

        cmd.append(url)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
            output = result.stdout
            parts = output.rsplit("\n__HTTP_STATUS__", 1)
            body = parts[0] if len(parts) == 2 else output
            status = int(parts[1]) if len(parts) == 2 else 0
            
            if status != 200:
                logger.debug(f"Curl {method} {url} returned status {status}. Body: {body[:200]}...")
                
            return status, body
        except subprocess.TimeoutExpired:
            logger.error(f"Curl request to {url} timed out")
            return 0, ""
        except Exception as e:
            logger.error(f"Error executing curl: {e}")
            return 0, ""

    def get_fresh_cookies(self) -> bool:
        """Get fresh cookies by visiting the main page"""
        try:
            logger.info("Getting fresh cookies from niftyindices.com...")
            if os.path.exists(self.cookie_file):
                os.remove(self.cookie_file)
            status, _ = self._curl(f'{self.base_url}/reports/historical-data')
            if status == 200:
                logger.info("✓ Fresh cookies obtained successfully")
                return True
            else:
                logger.error(f"✗ Failed to get fresh cookies: HTTP {status}")
                return False
        except Exception as e:
            logger.error(f"✗ Error getting fresh cookies: {e}")
            return False

    def fetch_index_data(self, index_name: str, start_date: str = '01-Jan-2015', end_date: Optional[str] = None):
        """Fetch data for a specific index from the API."""
        if end_date is None:
            end_date = datetime.now().strftime('%d-%b-%Y')

        payload = {
            "cinfo": json.dumps({
                'name': index_name,
                'startDate': start_date,
                'endDate': end_date,
                'indexName': index_name
            })
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                status, body = self._curl(self.api_url, method="POST", data=payload, timeout=60)

                if status == 200:
                    try:
                        data = json.loads(body)
                        # The API returns data in the 'd' field as a JSON-encoded string often
                        if data.get('d') and data['d'] != '[]':
                            return data
                        else:
                            logger.warning(f"Empty data received for {index_name}, attempt {attempt + 1}")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON response for {index_name}")

                elif status == 500:
                    logger.error(f"Server error for {index_name}, attempt {attempt + 1}")

                else:
                    logger.warning(f"HTTP {status} for {index_name}, attempt {attempt + 1}")

                if attempt < max_retries - 1:
                    logger.info("  Refreshing cookies and retrying...")
                    self.get_fresh_cookies()
                    time.sleep(2)

            except Exception as e:
                logger.error(f"Request error for {index_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        return None

    def save_index_data(self, index_name: str, data: Dict) -> Dict:
        """Save index data to JSON file and return statistics."""
        try:
            filename = index_name.replace('/', '-').replace(' ', '_')
            filepath = self.data_dir / f"{filename}.json"

            new_count = 0
            if data and 'd' in data:
                d_content = data['d']
                if isinstance(d_content, list):
                    new_count = len(d_content)
                elif isinstance(d_content, str) and d_content != '[]':
                    try:
                        parsed_d = json.loads(d_content)
                        if isinstance(parsed_d, list):
                            new_count = len(parsed_d)
                        else:
                            new_count = 1
                    except:
                        new_count = 1

            old_count = 0
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if existing_data and 'd' in existing_data:
                            old_d = existing_data['d']
                            if isinstance(old_d, list):
                                old_count = len(old_d)
                            elif isinstance(old_d, str) and old_d != '[]':
                                try:
                                    old_parsed = json.loads(old_d)
                                    if isinstance(old_parsed, list):
                                        old_count = len(old_parsed)
                                    else:
                                        old_count = 1
                                except:
                                    old_count = 1
                except:
                    old_count = 0

            diff = new_count - old_count
            logger.info(f"  📊 {index_name} count: {old_count} → {new_count} ({'+' if diff >= 0 else ''}{diff})")

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return {
                'success': True,
                'old_count': old_count,
                'new_count': new_count,
                'change': diff,
                'filepath': str(filepath)
            }

        except Exception as e:
            logger.error(f"  ✗ Error saving data for {index_name}: {e}")
            return {'success': False}

    def process_data_to_dataframe(self, data: Dict) -> Optional[Any]:
        """Convert API JSON data to a pandas DataFrame for analysis."""
        try:
            import pandas as pd
            if not data or 'd' not in data:
                return None
            
            d_content = data['d']
            if isinstance(d_content, str):
                records = json.loads(d_content)
            else:
                records = d_content
                
            if not records:
                return None
                
            df = pd.DataFrame(records)
            
            # Convert Date strings to datetime objects
            # Format is typically '11 Mar 2026'
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
            # Convert values to numeric
            for col in ['TotalReturnsIndex', 'NTR_Value', 'Index Value']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df
        except Exception as e:
            logger.error(f"Error converting to DataFrame: {e}")
            return None

def main():
    """Simple test function."""
    logging.basicConfig(level=logging.INFO)
    fetcher = NiftyIndexFetcher()
    if fetcher.get_fresh_cookies():
        data = fetcher.fetch_index_data("Nifty 50")
        if data:
            fetcher.save_index_data("Nifty 50", data)
            df = fetcher.process_data_to_dataframe(data)
            if df is not None:
                print(df.tail())

if __name__ == "__main__":
    main()
