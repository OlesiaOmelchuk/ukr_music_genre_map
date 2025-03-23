from time import sleep
import random
from loguru import logger

import requests
import atexit
import urllib.parse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Initialize Selenium WebDriver
options = Options()
options.add_argument("--headless")  # Run in headless mode for speed
options.add_argument("--disable-blink-features=AutomationControlled")
driver = webdriver.Chrome(options=options)

# Close the driver when the script exits
atexit.register(driver.quit)


def get_soup(url):
    sleep(random.uniform(0.2, 0.6))  # Small random delay to avoid rate limiting

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200 or response.text.strip() == "":
        logger.warning(f"Fallback to Selenium for {url}")
        driver.get(url)
        html = driver.page_source
    else:
        html = response.text
    return BeautifulSoup(html, "html.parser")

def soup_select_wrapper(soup, selector):
    try:
        return soup.select(selector)
    except Exception as e:
        logger.error(f"Error selecting {selector}: {e}")
        return None

def construct_lastfm_url(artist, title):
    base_url = "https://www.last.fm/music"

    # Replace spaces with + for both artist and title
    artist_formatted = urllib.parse.quote(artist.replace(" ", "+"), safe="+")
    title_formatted = urllib.parse.quote(title.replace(" ", "+"), safe="+")  # Keep `+`, encode `/`

    return f"{base_url}/{artist_formatted}/_/{title_formatted}"