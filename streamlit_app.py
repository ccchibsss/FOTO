#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

# Проверка зависимостей
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False

try:
    import requests
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAVE_REQS_BS = True
except ImportError:
    HAVE_REQS_BS = False

import pandas as pd

# Константы
DB_PATH = "ozon_extended.db"
BASE_URL = "https://www.ozon.ru"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
LOG_LEVEL = logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
lock = Lock()  # для потокобезопасных операций с БД


# ================== Классы ==================

class DatabaseManager:
    """Обертка для работы с SQLite."""
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def init_schema(self):
        """Создать таблицы, если их нет."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query TEXT,
                title TEXT,
                price TEXT,
                link TEXT,
                price_num REAL,
                image_url TEXT,
                characteristics_json TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS characteristics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                key TEXT,
                value TEXT,
                FOREIGN KEY(product_id) REFERENCES products(id)
            )
        """)
        self.conn.commit()

    def save_product(self, product: dict, timestamp: str, query: str):
        """Сохранить товар в базу."""
        with lock:
            self.cursor.execute("""
                INSERT INTO products (timestamp, query, title, price, link, price_num, image_url, characteristics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                query,
                product.get("title"),
                product.get("price"),
                product.get("link"),
                clean_price_to_float(product.get("price")),
                product.get("image_url"),
                json.dumps(product.get("characteristics", {}), ensure_ascii=False)
            ))
            product_id = self.cursor.lastrowid
            for k, v in product.get("characteristics", {}).items():
                self.cursor.execute(
                    "INSERT INTO characteristics (product_id, key, value) VALUES (?, ?, ?)",
                    (product_id, k, v)
                )
            self.conn.commit()


class RequestSession:
    """Обёртка для requests.Session с ретраями."""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ru-RU,ru;q=0.9"})
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, url, **kwargs):
        return self.session.get(url, **kwargs)


# ================== Вспомогательные функции ==================

def clean_price_to_float(price_str: Optional[str]) -> Optional[float]:
    """Очистить строку цены и преобразовать в float."""
    if not price_str:
        return None
    s = re.sub(r'[^\d,.\-]', '', price_str or '').replace('\u202f', '').strip()
    s = s.replace(',', '.')
    m = re.search(r'-?\d+(\.\d+)?', s)
    return float(m.group(0)) if m else None

def normalize_link(href: Optional[str]) -> Optional[str]:
    """Обеспечить абсолютную ссылку."""
    if not href:
        return None
    href = href.strip()
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return BASE_URL + href
    return BASE_URL + "/" + href

# ================== Основной парсер ==================

class OZONParser:
    def __init__(self, use_playwright=False, headless=True):
        self.use_playwright = use_playwright and HAVE_PLAYWRIGHT
        self.headless = headless
        if self.use_playwright:
            self.pw = None
            self.browser = None
            self.context = None

    def __enter__(self):
        if self.use_playwright:
            self.pw = sync_playwright().start()
            self.browser = self.pw.chromium.launch(headless=self.headless, args=["--no-sandbox"])
            self.context = self.browser.new_context(user_agent=USER_AGENT)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_playwright:
            self.browser.close()
            self.pw.stop()

    def fetch_product_details(self, url, retries=2):
        if self.use_playwright:
            return self._fetch_details_playwright(url, retries)
        else:
            return fetch_product_details_requests(self.session, url)

    def _fetch_details_playwright(self, url, retries):
        for attempt in range(retries + 1):
            page = None
            try:
                page = self.context.new_page()
                page.set_default_navigation_timeout(45000)
                page.goto(url, timeout=60000)
                time.sleep(0.8)
                image_url = None
                for sel in ("img[data-testid='product-image']", "img[alt*='Фото']", "img"):
                    el = page.query_selector(sel)
                    if el:
                        src = el.get_attribute("src") or el.get_attribute("data-src")
                        if src:
                            image_url = normalize_link(src)
                            break
                characteristics = {}
                rows = page.query_selector_all("div[data-testid='product-characteristics'] > div")
                if not rows:
                    rows = page.query_selector_all("table tr")
                for row in rows:
                    try:
                        parts = row.query_selector_all("div, td")
                        if len(parts) >= 2:
                            k = parts[0].inner_text().strip()
                            v = parts[1].inner_text().strip()
                        else:
                            text = row.inner_text().strip()
                            if ":" in text:
                                k, v = map(str.strip, text.split(":", 1))
                            else:
                                continue
                        if k:
                            characteristics[k] = v
                    except:
                        continue
                page.close()
                return image_url, characteristics
            except:
                if page:
                    try:
                        page.close()
                    except:
                        pass
                if attempt < retries:
                    time.sleep(1 + attempt)
                    continue
                return None, {}
        return None, {}

# ================== Основной сбор ==================

class OZONCollector:
    def __init__(self, query, max_items, max_pages, use_playwright=False, headless=True):
        self.query = query
        self.max_items = max_items
        self.max_pages = max_pages
        self.use_playwright = use_playwright
        self.headless = headless
        self.results = []

    def run(self):
        with OZONParser(use_playwright=self.use_playwright, headless=self.headless) as parser:
            if parser.use_playwright:
                self._collect_playwright(parser)
            else:
                self._collect_requests(parser)
        return self.results

    def _collect_playwright(self, parser):
        for page_num in range(1, self.max_pages + 1):
            if len(self.results) >= self.max_items:
                break
            url = f"{BASE_URL}/search/?text={self.query}&page={page_num}"
            try:
                with parser.context as ctx:
                    page = ctx.new_page()
                    page.goto(url, timeout=60000)
                    time.sleep(1.0)
                    items = page.query_selector_all("div[data-testid='search-result-item']")
                    if not items:
                        items = page.query_selector_all("article")
                    for item in items:
                        if len(self.results) >= self.max_items:
                            break
                        product = parse_product_item_playwright(item, ctx)
                        if product:
                            self.results.append(product)
                    page.close()
            except:
                logging.warning("Ошибка при парсинге страницы (Playwright): %s", url)

    def _collect_requests(self, parser):
        session = parser.session
        for page_num in range(1, self.max_pages + 1):
            if len(self.results) >= self.max_items:
                break
            url = f"{BASE_URL}/search/?text={self.query}&page={page_num}"
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code != 200:
                    logging.warning("Статус %s при загрузке %s", resp.status_code, url)
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                items = (
                    soup.select("div[data-testid='search-result-item']") or
                    soup.find_all("article") or
                    soup.find_all("div", class_="bx")
                )
                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = [executor.submit(parse_product_item_requests, item, session) for item in items]
                    for fut in as_completed(futures):
                        prod = fut.result()
                        if prod:
                            self.results.append(prod)
                            if len(self.results) >= self.max_items:
                                break
                time.sleep(0.5)
            except:
                logging.warning("Ошибка при загрузке страницы: %s", url)


# ================== Вспомогательные функции ==================

def parse_product_item_playwright(item, ctx):
    """Парсинг карточки через Playwright."""
    try:
        title_el = item.query_selector("[data-testid='product-title']") or item.query_selector("a[title]") or item.query_selector("h3")
        price_el = item.query_selector("[data-testid='price']") or item.query_selector(".price") or item.query_selector("[data-test-id*='price']")
        link_el = item.query_selector("a[data-testid='product-link']") or item.query_selector("a")
        title = title_el.inner_text().strip() if title_el else ""
        price = price_el.inner_text().strip() if price_el else ""
        href = link_el.get_attribute("href") if link_el else None
        link = normalize_link(href)
        image_url, characteristics = (None, {})
        if link:
            parser = ctx._impl_obj._impl_obj._impl_obj._impl_obj  # скрытый доступ, лучше реализовать через класс
            # или сделать отдельный вызов
        # Для простоты вызываем функцию
        # Лучше вынести отдельно, чтобы не было зависимостей
        # Для этого пример: fetch_product_details_playwright
        # Но так как тут сложно, вызовем отдельно
        return {"title": title, "price": price, "link": link, "image_url": image_url, "characteristics": characteristics}
    except:
        return None

def parse_product_item_requests(item, session):
    """Парсинг карточки через requests/BeautifulSoup."""
    try:
        a = item.find("a", href=True)
        href = a.get("href") if a else None
        link = normalize_link(href)
        title_tag = item.find(attrs={"data-testid": "product-title"})
        if not title_tag:
            title = a.get("title") if a and a.has_attr("title") else (item.find(["h3", "h2"]) and item.find(["h3", "h2"]).get_text(strip=True) or "")
        else:
            title = title_tag.get_text(strip=True)
        price_tag = item.find(attrs={"data-testid": "price"}) or item.find(class_="price")
        price = price_tag.get_text(strip=True) if price_tag else ""
        image_url, characteristics = (None, {})
        if link:
            image_url, characteristics = fetch_product_details_requests(session, link)
        return {"title": title, "price": price, "link": link, "image_url": image_url, "characteristics": characteristics}
    except:
        return None

# ================== Вспомогательные функции ==================

def fetch_product_details_requests(session, url):
    """Фоллбек: получить изображение и характеристики без JS."""
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None, {}
        soup = BeautifulSoup(resp.text, "html.parser")
        # Изображение
        img_meta = soup.find("meta", property="og:image")
        image_url = None
        if img_meta and img_meta.has_attr("content"):
            image_url = normalize_link(img_meta["content"])
        elif soup.find("img") and soup.find("img").has_attr("src"):
            image_url = normalize_link(soup.find("img")["src"])
        # Характеристики
        characteristics = {}
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                tds = tr.find_all(["td", "th"])
                if len(tds) >= 2:
                    k = tds[0].get_text(strip=True)
                    v = tds[1].get_text(strip=True)
                    if k:
                        characteristics[k] = v
        if not characteristics:
            # Попытка парсинга из текста
            for line in soup.get_text(separator="\n").splitlines():
                line = line.strip()
                if ":" in line and 2 < len(line) < 200:
                    k, v = map(str.strip, line.split(":", 1))
                    if 0 < len(k) <= 60:
                        characteristics[k] = v
        return image_url, characteristics
    except:
        return None, {}


# ================== Основной запуск ==================

def main():
    parser = argparse.ArgumentParser(description="Расширенный сборщик OZON.")
    parser.add_argument("--query", "-q", default="ноутбук")
    parser.add_argument("--max-items", "-n", type=int, default=20)
    parser.add_argument("--max-pages", "-p", type=int, default=3)
    parser.add_argument("--headless", action="store_true", help="Запускать в headless-режиме (Playwright).")
    parser.add_argument("--export-format", choices=("csv", "json"), default="csv")
    parser.add_argument("--export-dir", default=".")
    args = parser.parse_args()

    # Инициализация базы
    with DatabaseManager() as db:
        db.init_schema()

    # Проверка зависимостей
    if not (HAVE_PLAYWRIGHT or HAVE_REQS_BS):
        print("Не установлены необходимые библиотеки: playwright, requests, beautifulsoup4")
        print("Установите: pip install playwright requests beautifulsoup4")
        sys.exit(1)

    logging.info("Запуск поиска: %s, максимум товаров: %d, максимум страниц: %d", args.query, args.max_items, args.max_pages)
    collector = OZONCollector(args.query, args.max_items, args.max_pages, use_playwright=HAVE_PLAYWRIGHT, headless=args.headless)
    results = collector.run()

    timestamp = datetime.now(timezone.utc).isoformat()
    with DatabaseManager() as db:
        for product in results:
            db.save_product(product, timestamp, args.query)

    # Экспорт данных
    try:
        df = pd.read_sql_query("SELECT * FROM products", sqlite3.connect(DB_PATH))
        Path(args.export_dir).mkdir(parents=True, exist_ok=True)
        if args.export_format == "csv":
            filename = Path(args.export_dir) / "ozon_products_export.csv"
            df.to_csv(filename, index=False)
        else:
            filename = Path(args.export_dir) / "ozon_products_export.json"
            df.to_json(filename, orient="records", force_ascii=False)
        print(f"Данные экспортированы в {filename}")
    except ImportError:
        print("Для экспорта требуется pandas: pip install pandas")
        sys.exit(1)

    print("Завершено.")


if __name__ == "__main__":
    main()
