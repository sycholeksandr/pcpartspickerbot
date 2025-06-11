# 🛠️ PCPartBot — AI-Powered PC Build Recommender with Telegram Bot & Web Scraper

**PCPartBot** is an intelligent system for parsing, analyzing, and recommending custom PC builds based on real-world data and user requirements.  
It combines **web scraping**, **natural language processing**, **machine learning**, and a **Telegram bot interface** to deliver tailored hardware recommendations.

---

## 📌 Key Features

- 🔍 **Web Scraper**  
  Extracts real PC builds from [pcpartpicker.com](https://pcpartpicker.com/) using undetected ChromeDriver and Selenium.

- 🤖 **NLP Integration**  
  Uses a large language model (via Groq API) to extract user intent — including budget and usage scenario — from free-form text.

- 🧠 **PC Recommendation Model**  
  A trained neural network (`PCBuildModel`) predicts optimal hardware combinations for gaming or productivity tasks.

- 💬 **Telegram Bot Interface**  
  Enables users to interact via chat and receive instant build recommendations based on their needs.

- 📊 **Data Pipeline**  
  Includes data scraping, parsing, cleaning, encoding, and training over real-world PC build datasets.

- 🔒 **Configurable Settings**  
  Secrets like API keys and model paths are stored securely in a separate configuration module.

---

## 🗂️ Project Structure

pcpartbot/
├── bot/ # Telegram bot logic and handlers
├── config/ # Configuration (API keys, model paths) — not versioned
├── data/ # Raw and processed datasets, analysis scripts
├── model/ # Model definition, training, and inference
├── nlp_integration/ # Groq-based NLP parsing (intent extraction)
├── scraper/ # Web scraper for real PCPartPicker builds
└── main.py # Script entry point (build recommendation + scraping)


---

## ⚙️ Technologies Used

- **Python 3.10+**
- [PyTorch](https://pytorch.org/) — model training & inference  
- [Selenium](https://www.selenium.dev/) + [undetected-chromedriver](https://github.com/ultrafunkamsterdam/undetected-chromedriver) — data scraping  
- [Groq API](https://groq.com/) + [LLaMA 3](https://ai.meta.com/llama/) — LLM-based NLP  
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) — bot framework  
- [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/stable/) — data preprocessing & encoding  
- [Matplotlib](https://matplotlib.org/) — optional for visualization

---

## 💡 Example Use Case

1. A user opens the Telegram bot and types:
    I have $2600 for a new gaming PC
    
2. The NLP module extracts:
- **Budget:** $2600
- **Task:** Gaming

3. The ML model selects the best-fit components from learned data and returns a recommended build:
CPU: Amd Ryzen 7 9800X3D
Motherboard: X870
Memory: 64GB
Video Card: RX7900
Power Supply: 850W

---

## 🚀 Getting Started

> ⚠️ This project requires Python 3.10+ and access to Groq's LLM API.

```bash
# Clone the repo
git clone https://github.com/yourusername/pcpartbot.git
cd pcpartbot

# Install dependencies
pip install -r requirements.txt

# Add your configuration in config/settings.py (not tracked by Git)

# Run bot or scrape data
python -m bot.main.py           # Launch Telegram bot
python -m scraper.pcpart_scraper.py   # Scrape builds

📄 License
This project is open-source and licensed under the MIT License.