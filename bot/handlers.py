"""Handlers for Telegram bot callbacks and messages."""

from telegram import Update
from telegram.ext import ContextTypes

from .keyboards import start_keyboard
from .recommender import recommend_parts
from nlp_integration.nlp import extract_price_task, generate_recommendations


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /start command from the user.

    Args:
        update (Update): Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): Context object with user data.
    """
    context.user_data["active"] = False
    await update.message.reply_text(
        "Привіт! Натисни '🚀 Почати', щоб розпочати підбір конфігурації.",
        reply_markup=start_keyboard
    )


async def start_build(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Activate build process after '🚀 Почати' is pressed.

    Args:
        update (Update): Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): Context object with user data.
    """
    context.user_data["active"] = True
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(
        "👷 Підбір увімкнено. Напиши свій бюджет і тип задач (ігри чи робота)."
    )


async def stop_build(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Deactivate build process after '🛑 Зупинити' is pressed.

    Args:
        update (Update): Telegram update object.
        context (ContextTypes.DEFAULT_TYPE): Context object with user data.
    """
    context.user_data["active"] = False
    await update.callback_query.answer()
    await update.callback_query.edit_message_text("🛑 Підбір вимкнено.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle user text input and generate PC configuration.

    Args:
        update (Update): Telegram update object containing the message.
        context (ContextTypes.DEFAULT_TYPE): Context with user state.
    """
    user_text = update.message.text.strip().lower()

    if user_text == "🚀 почати":
        context.user_data["active"] = True
        await update.message.reply_text(
            "👷 Напиши свій бюджет($) і тип задач (ігри чи робота)."
        )
        return

    if user_text == "🛑 зупинити":
        context.user_data["active"] = False
        await update.message.reply_text(
            "🛑 Підбір завершено. Натисни '🚀 Почати', щоб розпочати знову."
        )
        return

    if not context.user_data.get("active", False):
        await update.message.reply_text("⚠️ Натисни '🚀 Почати', щоб розпочати.")
        return

    data = await extract_price_task(user_text)
    if not data:
        await update.message.reply_text(
            "⚠️ Не вдалося зрозуміти запит. Наприклад: 'ПК до 1200$ для ігор'."
        )
        return

    build = recommend_parts(data["price"], data["task"])
    recommendations = await generate_recommendations(build)

    response_text = (
        "Ось твоя рекомендована збірка за запитом:\n"
        + "\n".join(f"{k}: {v}" for k, v in build.items())
        + "\n\n📝 Рекомендації:\n"
        + recommendations
    )

    await update.message.reply_text(response_text)
    await update.message.reply_text(
        "🔁 Хочеш зібрати ще одну конфігурацію? Просто напиши новий запит або натисни '🛑 Зупинити'.",
        reply_markup=start_keyboard
    )
