"""Simple script to test the Telegram bot connection.
"""

import os
import asyncio
from telegram import Bot
from dotenv import load_dotenv

async def test_bot():
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        print("❌ Error: TELEGRAM_BOT_TOKEN not found in .env")
        return

    print(f"🤖 Testing Bot with token: {token[:10]}...")
    bot = Bot(token=token)
    
    try:
        me = await bot.get_me()
        print(f"✅ Success! Bot is online.")
        print(f"Name: {me.first_name}")
        print(f"Username: @{me.username}")
        print("\nNow start the interactive bot by running:")
        print("python -m src.notifications.telegram_bot")
        
    except Exception as e:
        print(f"❌ Failed to connect to Telegram: {e}")

if __name__ == "__main__":
    asyncio.run(test_bot())
