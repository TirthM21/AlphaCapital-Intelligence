import logging
import os
import requests
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
        """Format and send a list of signals."""
        if not signals:
            return
            
        msg = f"🚀 *{title} Signal Alert*\n\n"
        
        for sig in signals[:15]: # Limit to top 15 to avoid telegram message size limits
            msg += f"• *{sig['ticker']}*: {sig['price']} (RSI: {sig['rsi']})\n"
            msg += f"  Signals: {', '.join(sig['signals'])}\n"
            msg += "\n"
            
        if len(signals) > 15:
            msg += f"\n... and {len(signals) - 15} more."
            
        self.send_message(msg)
