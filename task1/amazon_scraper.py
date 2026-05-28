import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import random

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

BASE_URL = "https://www.amazon.in/s?k={query}&page={page}"


def human_delay(is_page_turn=False):
    if is_page_turn:
        delay = random.uniform(2.0, 3.0)
        print(f"  waiting {delay:.1f}s...")
        time.sleep(delay)
    else:
        time.sleep(random.uniform(0.4, 1.2))


def get_soup(url, is_page_turn=False):
    human_delay(is_page_turn=is_page_turn)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def is_sponsored(card):
    badge = card.find("span", string=lambda t: t and "Sponsored" in t)
    if badge:
        return "Ad"
    label = card.find("span", {"class": lambda c: c and "sponsored" in c.lower()})
    return "Ad" if label else "Organic"


def parse_card(card):
    title_block = card.find("h2")
    
    if title_block:
        title_tag = title_block.find("span", {"class": "a-text-normal"})
        if title_tag:
            title = title_tag.get_text(strip=True)
        else:
            title = title_block.get_text(strip=True)
    else:
        title_tag = card.find("span", {"class": "a-text-normal"})
        title = title_tag.get_text(strip=True) if title_tag else None

    if title:
        words = title.split()  
        if len(words) > 8:     
            title = " ".join(words[:8]) + "..."  

    img_tag = card.find("img", {"class": "s-image"})
    image = img_tag["src"] if img_tag and img_tag.get("src") else None

    rating_tag = card.find("span", {"class": "a-icon-alt"})
    rating = rating_tag.get_text(strip=True) if rating_tag else None

    price_whole = card.find("span", {"class": "a-price-whole"})
    price_fraction = card.find("span", {"class": "a-price-fraction"})
    if price_whole:
        whole = price_whole.get_text(strip=True).replace(",", "").replace(".", "")
        fraction = price_fraction.get_text(strip=True) if price_fraction else "00"
        price = f"₹{whole}.{fraction}"
    else:
        price = None

    result_type = is_sponsored(card)

    return {
        "title": title,
        "image_url": image,
        "rating": rating,
        "price": price,
        "result_type": result_type,
    }


def scrape_products(query = "laptop",pages=5):
    all_products = []

    for page in range(1, pages + 1):
        print(f"Scraping page {page}...")
        url = BASE_URL.format(query = query,page=page)

        try:
            soup = get_soup(url, is_page_turn=(page > 1))
        except Exception as e:
            print(f"  Failed page {page}: {e}")
            continue

        cards = soup.find_all("div", {"data-component-type": "s-search-result"})
        print(f"  Found {len(cards)} products")

        for card in cards:
            product = parse_card(card)
            if product["title"]:
                all_products.append(product)

    return all_products


def save_to_csv(products):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"amazon_laptops_{timestamp}.csv"
    df = pd.DataFrame(products)
    df.drop_duplicates(subset=["title"], inplace=True)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} products → {filename}")
    return filename


if __name__ == "__main__":
    products = scrape_products(pages=5)
    if products:
        save_to_csv(products)
    else:
        print("No products scraped. Amazon may be blocking requests.")