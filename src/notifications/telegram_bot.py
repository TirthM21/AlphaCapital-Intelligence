"""On-demand Telegram Bot for stock screening.

Allows users to request scans and analysis directly from Telegram.
"""

import os
import logging
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import pandas as pd
import yfinance as yf
from src.screening.extended_signals import score_extended_signals
from src.data.universe_fetcher import StockUniverseFetcher

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Token from .env
from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command help."""
    await update.message.reply_text(
        "Welcome to the PKScreener Clone Bot! 🚀\n\n"
        "Commands:\n"
        "/scan [INDEX] - Scan an index (e.g. /scan NIFTY 50)\n"
        "/signals [TICKER] - Get detailed signals for a stock\n"
        "/ipo - Scan recent IPOs\n"
        "/fno - Scan F&O stocks"
    )

async def scan_index(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /scan command."""
    index_name = " ".join(context.args) if context.args else "NIFTY 50"
    await update.message.reply_text(f"🔍 Scanning {index_name}... Please wait.")
    
    try:
        fetcher = StockUniverseFetcher()
        tickers = fetcher.fetch_universe(index_name=index_name)[:20] # Limit for Telegram 
        
        results = []
        for t in tickers:
            df = yf.download(t, period="1y", interval="1d", progress=False)
            if not df.empty:
                res = score_extended_signals(t, df)
                if res.get('signals'):
                    results.append(res)
                    
        if not results:
            await update.message.reply_text("No signals found for this index today.")
            return

        report = f"📊 Scan Results for {index_name}:\n\n"
        for r in results[:10]:
            report += f"🔹 *{r['ticker']}* - {r['price']}\n"
            report += f"Signals: {', '.join(r['signals'][:3])}\n"
            report += f"AI Score: {r['prediction_score']}/100\n\n"
            
        await update.message.reply_text(report, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Scan error: {e}")
        await update.message.reply_text("Error occurred during scan.")

async def stock_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /signals [TICKER] command."""
    if not context.args:
        await update.message.reply_text("Please provide a ticker. Example: /signals RELIANCE")
        return
        
    ticker = context.args[0].upper()
    if "." not in ticker: ticker += ".NS"
    
    await update.message.reply_text(f"📖 Analyzing {ticker}...")
    
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty:
            await update.message.reply_text("Could not find data for this stock.")
            return
            
        res = score_extended_signals(ticker, df)
        
        msg = f"🔍 *Analysis for {ticker}*\n"
        msg += f"Price: {res['price']} | RSI: {res['rsi']}\n"
        msg += f"Lorentzian: {res['lorentzian']}\n"
        msg += f"AI Score: {res['prediction_score']}/100\n"
        msg += f"Forecast: {res['next_day_forecast']}\n\n"
        msg += "*Signals:*\n"
        for s in res['signals']:
            msg += f"✅ {s}\n"
            
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Signal error: {e}")
        await update.message.reply_text("Error analyzing this stock.")

def run_bot():
    """Run the bot polling loop."""
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found in .env")
        return
        
    app = ApplicationBuilder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan_index))
    app.add_handler(CommandHandler("signals", stock_signals))
    app.add_handler(CommandHandler("ipo", lambda u, c: scan_index(u, c))) # Example
    
    print("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    run_bot()
