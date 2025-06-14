"""Entry point of the Telegram bot."""
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters
)
from config.settings import TELEGRAM_TOKEN
from bot.handlers import start, start_build, stop_build, handle_message
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Start the bot."""
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(start_build, pattern="^start_build$"))
    app.add_handler(CallbackQueryHandler(stop_build, pattern="^stop_build$"))

    app.run_polling()

if __name__ == "__main__":
    main()
