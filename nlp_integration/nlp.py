"""NLP integration with Groq API (LLaMA3)."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import groq
from config.settings import GROQ_API_KEY
import logging

logger = logging.getLogger(__name__)
CLIENT = groq.Groq(api_key=GROQ_API_KEY)
EXECUTOR = ThreadPoolExecutor()

async def extract_price_task(text: str):
    """Extract budget and task type from user input."""
    prompt = (
        f'"{text}"\n'
        'Витягни тільки бюджет (число з грошовою одиницею або без) і тип задачі (ігри або робота) і поверни **тільки** JSON з цими даними без пояснень.\n'
        'Формат JSON: {"price": <float>, "task": "games" або "work"}\n'
        'Якщо не вдалося — {"price": null, "task": null}'
    )

    def request():
        response = CLIENT.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        logger.info(f"NLP raw response: {response.choices[0].message.content!r}")
        return response.choices[0].message.content

    result = await asyncio.get_event_loop().run_in_executor(EXECUTOR, request)
    return json.loads(result)

async def generate_recommendations(build: dict) -> str:
    """Generate LLM-based recommendations for given PC build."""
    prompt = (
        f"Збірка:\n"
        f"CPU: {build['CPU']}\n"
        f"Motherboard: {build['Motherboard']}\n"
        f"Memory: {build['Memory']}\n"
        f"Video Card: {build['Video Card']}\n"
        f"Power Supply: {build['Power Supply']}\n\n"
        "Дай короткі технічні рекомендації. Формат:\n"
        "- Кулер: TDP + Socket of the Motherboard, usually: AM4, AM5 LGA1700 etc.\n"
        "- Корпус: форм-фактор материнської плати\n"
        "- Вентилятори: скільки потрібно\n"
        "- PSU сертифікація: рекомендована\n\n"
        "Жодних пояснень. Лише список."
    )

    def request():
        response = CLIENT.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

    return await asyncio.get_event_loop().run_in_executor(EXECUTOR, request)
