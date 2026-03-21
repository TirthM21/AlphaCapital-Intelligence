import pandas as pd
import requests
import logging
from io import StringIO
from typing import List, Dict, Optional
from nse import NSE
from pathlib import Path

logger = logging.getLogger(__name__)

class NSEFetcher:
    """Enhanced NSE data fetcher using the official-yet-unofficial nse library."""
    
    def __init__(self, download_folder: str = './data/nse_downloads'):
        self.download_folder = Path(download_folder)
        self.download_folder.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        self.fallback = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"]

    def get_all_equity_stocks(self) -> List[str]:
        """Fetch all equity symbols from NSE."""
        try:
            url = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                df = pd.read_csv(StringIO(res.text))
                if 'SYMBOL' in df.columns:
                    return df['SYMBOL'].tolist()
        except Exception as e:
            logger.error(f"Error fetching symbols from CSV: {e}")
            
        # Fallback to nse library
        try:
            with NSE(str(self.download_folder)) as nse:
                data = nse.listEquityStocksByIndex("NIFTY 500")
                if 'data' in data:
                    return [item['symbol'] for item in data['data']]
        except Exception as e:
            logger.error(f"Error fetching from nse library: {e}")
            
        return self.fallback

    def get_index_stocks(self, index_name: str) -> List[str]:
        """Fetch symbols belonging to a specific index."""
        try:
            with NSE(str(self.download_folder)) as nse:
                data = nse.listEquityStocksByIndex(index_name)
                if 'data' in data:
                    return [item['symbol'] for item in data['data']]
        except Exception as e:
            logger.error(f"Error fetching {index_name} from nse: {e}")
        return []

    def get_etfs(self) -> List[str]:
        """Fetch all ETFs."""
        try:
            with NSE(str(self.download_folder)) as nse:
                data = nse.listEtf()
                if 'data' in data:
                    return [item['symbol'] for item in data['data']]
        except Exception as e:
            logger.error(f"Error fetching ETFs: {e}")
        return ["NIFTYBEES", "BANKBEES", "GOLDBEES"]

    def get_ipos(self, type: str = 'current') -> List[str]:
        """Fetch IPO information and return a list of symbols."""
        try:
            with NSE(str(self.download_folder)) as nse:
                if type == 'current':
                    data = nse.listCurrentIPO()
                elif type == 'upcoming':
                    data = nse.listUpcomingIPO()
                elif type == 'past':
                    import datetime
                    fromdate = datetime.datetime.now() - datetime.timedelta(days=730) # Last 2 years
                    todate = datetime.datetime.now()
                    data = nse.listPastIPO(from_date=fromdate, to_date=todate)
                else:
                    data = []
                    
                if data:
                    # nse library returns different keys based on type
                    symbols = []
                    for item in data:
                        # Try commonly used keys for symbol
                        sym = item.get('symbol') or item.get('issueSymbol') or item.get('companyName')
                        if sym: symbols.append(sym)
                    return symbols
        except Exception as e:
            logger.error(f"Error fetching IPOs: {e}")
        return []

    def get_fno_stocks(self) -> List[str]:
        """Fetch all F&O stocks."""
        try:
            with NSE(str(self.download_folder)) as nse:
                # Based on documentation nse.listFnoStocks() is deprecated, 
                # use listEquityStocksByIndex(index='SECURITIES IN F&O')
                data = nse.listEquityStocksByIndex('SECURITIES IN F&O')
                if 'data' in data:
                    return [item['symbol'] for item in data['data']]
        except Exception as e:
            logger.error(f"Error fetching F&O stocks: {e}")
        return self.fallback

    def get_market_status(self) -> List[Dict]:
        """Fetch current market status."""
        try:
            with NSE(str(self.download_folder)) as nse:
                return nse.status()
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
        return []

    def get_index_list(self) -> Dict:
        """Fetch list of all indices."""
        try:
            with NSE(str(self.download_folder)) as nse:
                return nse.listIndices()
        except Exception as e:
            logger.error(f"Error fetching index list: {e}")
        return {}
