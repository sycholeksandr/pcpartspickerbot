"""Keyboard markup for the Telegram bot."""
from telegram import ReplyKeyboardMarkup, KeyboardButton

start_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("🚀 Почати"), KeyboardButton("🛑 Зупинити")]
    ],
    resize_keyboard=True,
    one_time_keyboard=False
)
