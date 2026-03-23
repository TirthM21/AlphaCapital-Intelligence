import logging
import os
import requests
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Sends notifications to Telegram channel/bot."""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
    def send_message(self, message: str) -> bool:
        """Send a simple text message."""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured. Skipping notification.")
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Telegram notification sent successfully.")
            return True
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False

    def send_signals(self, signals: list, title: str):
        """Format and send a list of signals with Buy/Stop/Target details if available."""
        if not signals:
            return
            
        msg = f"🚀 *{title} Signal Alert*\n"
        msg += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        # Sort by score if present
        if 'score' in signals[0]:
            signals = sorted(signals, key=lambda x: x.get('score', 0), reverse=True)
        elif 'alpha_score' in signals[0]:
            signals = sorted(signals, key=lambda x: x.get('alpha_score', 0), reverse=True)

        for sig in signals[:15]: # Limit to top 15
            ticker = sig.get('ticker', 'Unknown')
            price = sig.get('price', sig.get('current_price', 'N/A'))
            
            msg += f"• *{ticker}* | Price: {price}\n"
            
            # Case 1: Advanced Regime Signals
            if 'strategy' in sig and 'stop' in sig:
                msg += f"  Type: {sig['strategy']} (Score: {sig.get('score', 0):.2f})\n"
                msg += f"  🎯 Target: {sig.get('target', 'N/A')} | 🛑 Stop: {sig.get('stop', 'N/A')}\n"
                msg += f"  📦 Qty: {sig.get('quantity', 'N/A')} (Risk: {sig.get('risk_pct', 'N/A')}%)\n"
            
            # Case 2: Extended Technical Signals
            elif 'signals' in sig:
                msg += f"  Signals: {', '.join(sig['signals'][:4])}\n"
                if 'rsi' in sig: msg += f"  RSI: {sig['rsi']:.1f}\n"

            msg += "\n"
            
        if len(signals) > 15:
            msg += f"\n... and {len(signals) - 15} more stocks found."
            
        self.send_message(msg)
