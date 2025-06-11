import csv
import logging
import os
import random
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import undetected_chromedriver as uc
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    NoSuchElementException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === Constants ===
LOGGER = logging.getLogger()
SKIP_COMPONENTS = {
    "Case Fan", "Case", "Keyboard", "Storage", "Mouse", "Operating System",
    "Monitor", "Speakers", "Headphones", "Sound Card",
    "Wired Network Adapter", "Wireless Network Adapter"
}
MONITOR_WIDTH = 2560
MONITOR_HEIGHT = 1440
MAX_PAGES = 70
FILTER_PARAMS = "33,41,39,40,42,28,35,36,38"

# Fallback to user home directory for UC patched driver
UC_PATCHED_DRIVER_PATH = os.path.expanduser(
    "~/.undetected_chromedriver/undetected_chromedriver.exe"
)

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# === Utility Functions ===

def random_delay(min_seconds=2, max_seconds=5):
    """Sleep for a random number of seconds between min_seconds and max_seconds."""
    time.sleep(random.uniform(min_seconds, max_seconds))


def clean_chrome_profiles():
    """Delete any temporary Chrome profiles from previous sessions."""
    for i in range(4):
        profile_dir = f"./chrome_profile_{i}"
        if os.path.exists(profile_dir):
            try:
                shutil.rmtree(profile_dir)
                LOGGER.info(f"Deleted profile {profile_dir}")
            except OSError as e:
                LOGGER.warning(f"Could not delete {profile_dir}: {e}")


def get_driver(window_index=0):
    """
    Create a new Chrome driver instance using a temporary folder 
    and assign it a window position on screen based on index.
    """
    tmp_dir = tempfile.mkdtemp()
    new_driver_path = os.path.join(tmp_dir, "chromedriver.exe")
    shutil.copy(UC_PATCHED_DRIVER_PATH, new_driver_path)

    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-extensions")

    driver = uc.Chrome(
        options=options,
        driver_executable_path=new_driver_path,
        use_subprocess=True
    )

    width = MONITOR_WIDTH // 2
    height = MONITOR_HEIGHT // 2
    positions = [(0, 0), (width, 0), (0, height), (width, height)]
    x, y = positions[window_index % 4]
    driver.set_window_position(x, y)
    driver.set_window_size(width, height)

    return driver


def parse_price(price_str):
    """
    Convert a price string (e.g. "$123.45") into a float.
    Returns 0.0 if parsing fails.
    """
    try:
        return float(
            price_str.replace("$", "")
            .replace("£", "")
            .replace("AUD", "")
            .replace(",", "")
            .replace("CAD", "")
            .replace("€", "")
            .replace("EUR", "")
            .strip()
        )
    except ValueError:
        return 0.0


def process_page(page_num, window_index):
    """
    Scrape all builds from a specific page number.
    
    Args:
        page_num (int): Page number to process.
        window_index (int): Used for screen positioning of the Chrome window.

    Returns:
        list: List of builds, where each build is a list of component dictionaries.
    """
    LOGGER.info(f"Processing page {page_num}")
    try:
        driver = get_driver(window_index)
    except WebDriverException as e:
        LOGGER.error(f"Driver initialization failed: {e}")
        return []

    wait = WebDriverWait(driver, 30)
    builds = []
    page_url = f"https://pcpartpicker.com/builds/#s={FILTER_PARAMS}&page={page_num}"

    try:
        try:
            driver.get(page_url)
            wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "logGroup")))
        except TimeoutException:
            LOGGER.warning(f"Timeout loading page {page_num}. Retrying...")
            driver.refresh()
            wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "logGroup")))

        cards = driver.find_elements(By.CLASS_NAME, "logGroup")
        build_links = [
            card.find_element(By.CSS_SELECTOR, "a.logGroup__target").get_attribute("href")
            for card in cards
        ]

        for link in build_links:
            try:
                driver.get(link)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "partlist")))

                tbody = driver.find_element(By.CSS_SELECTOR, "tbody")
                rows = tbody.find_elements(By.TAG_NAME, "tr")

                current_category = None
                components = []
                skip_build = False

                for row in rows:
                    try:
                        category_cell = row.find_element(By.CLASS_NAME, "td__component")
                        if category_cell.get_attribute("colspan") == "2":
                            current_category = category_cell.find_element(By.TAG_NAME, "h4").text.strip()
                            continue
                    except NoSuchElementException:
                        pass

                    if current_category in SKIP_COMPONENTS:
                        continue

                    try:
                        name_cell = row.find_element(By.CLASS_NAME, "td__name")
                        price_cell = row.find_element(By.CLASS_NAME, "td__price")
                        name = name_cell.find_element(By.TAG_NAME, "a").text.strip()
                        price = price_cell.text.strip()
                        components.append({
                            "category": current_category,
                            "name": name,
                            "price": price
                        })
                    except NoSuchElementException:
                        skip_build = True
                        break

                if not skip_build and components:
                    builds.append(components)
                    LOGGER.info(f"Added build from {link}")

                random_delay(3, 7)

            except Exception as e:
                LOGGER.warning(f"Failed to process build {link}: {e}")
                continue

    except Exception as e:
        LOGGER.error(f"Critical error on page {page_num}: {e}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    return builds


def save_to_csv(all_builds, filename="parsed_builds5.csv"):
    """
    Save all scraped builds into a CSV file.

    Args:
        all_builds (list): List of build data.
        filename (str): Output CSV file name.
    """
    fieldnames = ['build_id', "CPU", "CPU Cooler", "Motherboard", "Memory", "Video Card", "Case", "Power Supply", "Total Price"]
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for build_id, components in enumerate(all_builds, start=1):
            row = {field: "" for field in fieldnames}
            row["build_id"] = build_id
            total_price = 0.0

            for comp in components:
                cat = comp['category']
                name = comp['name']
                price_str = comp['price']
                total_price += parse_price(price_str)

                if cat in fieldnames:
                    row[cat] = name

            row['Total Price'] = f"{total_price:.2f}"
            writer.writerow(row)


# === Main ===

if __name__ == "__main__":
    clean_chrome_profiles()

    all_builds = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(process_page, page_num, i % 3): page_num
            for i, page_num in enumerate(range(44, MAX_PAGES + 1))
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_builds.extend(result)
            except Exception as e:
                LOGGER.error(f"Thread error: {e}")

    save_to_csv(all_builds)
    LOGGER.info(f"Done! Saved {len(all_builds)} builds.")
