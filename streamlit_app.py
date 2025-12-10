Улучшённый и исправленный универсальный сборщик OZON (CLI).
- Исправлены ошибки: депрекация utcnow, некорректное использование Playwright в потоках,
  неправильные сообщения об ошибках, слабый requests-фоллбек.
- Если Playwright доступен — используется он (последовательный обход карточек для безопасности).
- Если Playwright недоступен — fallback requests+BeautifulSoup с ретраями через urllib3 Retry.
- Все поясняющие комментарии и логи — на русском.
"""
import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Опциональные зависимости
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    HAVE_PLAYWRIGHT = True
except Exception:
    HAVE_PLAYWRIGHT = False

try:
    import requests
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAVE_REQS_BS = True
except Exception:
    HAVE_REQS_BS = False

DB_PATH = "ozon_extended.db"
BASE_URL = "https://www.ozon.ru"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ----------------- БД -----------------
def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
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
    c.execute("""
        CREATE TABLE IF NOT EXISTS characteristics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            key TEXT,
            value TEXT,
            FOREIGN KEY(product_id) REFERENCES products(id)
        )
    """)
    conn.commit()
    conn.close()


# ----------------- Утилиты -----------------
def clean_price_to_float(price_str: Optional[str]) -> Optional[float]:
    if not price_str:
        return None
    s = re.sub(r'[^\d,.\-]', '', price_str or '').replace('\u202f', '').strip()
    s = s.replace(',', '.')
    m = re.search(r'-?\d+(\.\d+)?', s)
    return float(m.group(0)) if m else None


def normalize_link(href: Optional[str]) -> Optional[str]:
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


# ----------------- Детали товара -----------------
def fetch_product_details_playwright(context, product_url: str, retries: int = 2) -> Tuple[Optional[str], Dict[str, str]]:
    """Playwright: открыть страницу и извлечь image + характеристики (последовательно)."""
    for attempt in range(retries + 1):
        page = None
        try:
            page = context.new_page()
            page.set_default_navigation_timeout(45000)
            page.goto(product_url, timeout=60000)
            time.sleep(0.8)
            image_url = None
            for sel in ("img[data-testid='product-image']", "img[alt*='Фото']", "img"):
                el = page.query_selector(sel)
                if el:
                    src = el.get_attribute("src") or el.get_attribute("data-src")
                    if src:
                        image_url = normalize_link(src)
                        break
            characteristics: Dict[str, str] = {}
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
                except Exception:
                    continue
            page.close()
            return image_url, characteristics
        except Exception as e:
            logging.debug("Playwright detail fetch error (%s): %s", product_url, e)
            try:
                if page:
                    page.close()
            except Exception:
                pass
            if attempt < retries:
                time.sleep(1 + attempt)
                continue
            logging.warning("Не удалось получить детали товара (Playwright): %s", product_url)
            return None, {}
    return None, {}


def build_requests_session() -> requests.Session:
    """Создать requests.Session с retry/backoff."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ru-RU,ru;q=0.9"})
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_product_details_requests(session: requests.Session, product_url: str) -> Tuple[Optional[str], Dict[str, str]]:
    """Requests fallback: попытка получить image + характеристики без JS."""
    try:
        resp = session.get(product_url, timeout=15)
        if resp.status_code != 200:
            logging.debug("Requests detail status %s for %s", resp.status_code, product_url)
            return None, {}
        soup = BeautifulSoup(resp.text, "html.parser")
        image_tag = soup.find("meta", property="og:image") or soup.find("img")
        image_url = None
        if image_tag:
            if image_tag.has_attr("content"):
                image_url = normalize_link(image_tag["content"])
            elif image_tag.has_attr("src"):
                image_url = normalize_link(image_tag["src"])
        characteristics: Dict[str, str] = {}
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                tds = tr.find_all(["td", "th"])
                if len(tds) >= 2:
                    k = tds[0].get_text(strip=True)
                    v = tds[1].get_text(strip=True)
                    if k:
                        characteristics[k] = v
        if not characteristics:
            texts = soup.get_text(separator="\n").splitlines()
            for line in texts:
                line = line.strip()
                if ":" in line and 2 < len(line) < 200:
                    k, v = map(str.strip, line.split(":", 1))
                    if 0 < len(k) <= 60:
                        characteristics[k] = v
        return image_url, characteristics
    except Exception as e:
        logging.debug("Requests detail error %s: %s", product_url, e)
        return None, {}


# ----------------- Парс карточек -----------------
def parse_product_item_playwright(item, context) -> Optional[Dict]:
    """Извлечь title/price/link из Playwright-элемента, получить детали последовательно."""
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
            image_url, characteristics = fetch_product_details_playwright(context, link)
        return {"title": title, "price": price, "link": link, "image_url": image_url, "characteristics": characteristics}
    except Exception as e:
        logging.debug("parse item playwright error: %s", e)
        return None


def parse_product_item_requests(elem_soup, session: requests.Session) -> Optional[Dict]:
    """Парсер блока результата (requests fallback) — возвращает товар с деталями."""
    try:
        a = elem_soup.find("a", href=True)
        href = a.get("href") if a else None
        link = normalize_link(href)
        title_tag = elem_soup.find(attrs={"data-testid": "product-title"})
        if not title_tag:
            title = a.get("title") if a and a.has_attr("title") else (elem_soup.find(["h3", "h2"]).get_text(strip=True) if elem_soup.find(["h3", "h2"]) else "")
        else:
            title = title_tag.get_text(strip=True)
        price_tag = elem_soup.find(attrs={"data-testid": "price"}) or elem_soup.find(class_="price")
        price = price_tag.get_text(strip=True) if price_tag else ""
        image_url, characteristics = (None, {})
        if link:
            image_url, characteristics = fetch_product_details_requests(session, link)
        return {"title": title, "price": price, "link": link, "image_url": image_url, "characteristics": characteristics}
    except Exception as e:
        logging.debug("parse item requests error: %s", e)
        return None


# ----------------- Поиск -----------------
def fetch_search_results(query: str, max_items: int = 20, max_pages: int = 3, headless: bool = True) -> List[Dict]:
    """
    Выбор движка:
    - Playwright: последовательный обход карточек (без потоков чтобы не ломать Playwright)
    - Requests+BS: параллельный обход карточек (детали) с ретраями
    """
    results: List[Dict] = []
    if HAVE_PLAYWRIGHT:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless, args=["--no-sandbox"])
            context = browser.new_context(user_agent=USER_AGENT)
            try:
                for page_num in range(1, max_pages + 1):
                    if len(results) >= max_items:
                        break
                    url = f"{BASE_URL}/search/?text={query}&page={page_num}"
                    page = context.new_page()
                    try:
                        page.goto(url, timeout=60000)
                        time.sleep(1.0)
                        items = page.query_selector_all("div[data-testid='search-result-item']")
                        if not items:
                            items = page.query_selector_all("article")
                        # Последовательно парсим элементы (Playwright не потокобезопасен)
                        for item in items:
                            if len(results) >= max_items:
                                break
                            prod = parse_product_item_playwright(item, context)
                            if prod:
                                results.append(prod)
                        page.close()
                    except Exception as e:
                        logging.warning("Ошибка загрузки страницы (Playwright) %s: %s", url, e)
                        try:
                            page.close()
                        except Exception:
                            pass
                        time.sleep(1.0)
            finally:
                browser.close()
    else:
        # Requests fallback
        if not HAVE_REQS_BS:
            logging.error("Ни Playwright, ни requests+BeautifulSoup не доступны. Установите зависимости.")
            return results
        session = build_requests_session()
        for page_num in range(1, max_pages + 1):
            if len(results) >= max_items:
                break
            url = f"{BASE_URL}/search/?text={query}&page={page_num}"
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code != 200:
                    logging.warning("Статус %s при попытке загрузить %s", resp.status_code, url)
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                items = soup.select("div[data-testid='search-result-item']") or soup.find_all("article") or soup.find_all("div", class_="bx")
                # Параллельно парсим карточки (детали через requests тоже)
                with ThreadPoolExecutor(max_workers=6) as ex:
                    futures = [ex.submit(parse_product_item_requests, item, session) for item in items]
                    for fut in as_completed(futures):
                        prod = fut.result()
                        if prod:
                            results.append(prod)
                            if len(results) >= max_items:
                                break
                time.sleep(0.5)
            except Exception as e:
                logging.warning("Ошибка requests при загрузке %s: %s", url, e)
                time.sleep(0.5)
    return results[:max_items]


# ----------------- Сохранение и экспорт -----------------
def save_product_to_db(product: Dict, timestamp: str, query: str, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    price_num = clean_price_to_float(product.get("price", "")) if product.get("price") else None
    characteristics_json = json.dumps(product.get("characteristics", {}), ensure_ascii=False)
    c.execute("""
        INSERT INTO products (timestamp, query, title, price, link, price_num, image_url, characteristics_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, query, product.get("title"), product.get("price"), product.get("link"), price_num, product.get("image_url"), characteristics_json))
    product_id = c.lastrowid
    for k, v in (product.get("characteristics") or {}).items():
        c.execute("INSERT INTO characteristics (product_id, key, value) VALUES (?, ?, ?)", (product_id, k, v))
    conn.commit()
    conn.close()


def export_data(db_path: str = DB_PATH, export_dir: str = ".", fmt: str = "csv"):
    try:
        import pandas as pd
    except Exception:
        logging.error("Для экспорта требуется pandas. Установите: pip install pandas")
        return
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        out = Path(export_dir) / "ozon_products_export.csv"
        df.to_csv(out, index=False)
        logging.info("Экспортировано в %s", out)
    else:
        out = Path(export_dir) / "ozon_products_export.json"
        df.to_json(out, orient="records", force_ascii=False)
        logging.info("Экспортировано в %s", out)


# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="OZON crawler (Playwright если есть, иначе requests fallback).")
    parser.add_argument("--query", "-q", default="ноутбук")
    parser.add_argument("--max-items", "-n", type=int, default=20)
    parser.add_argument("--max-pages", "-p", type=int, default=3)
    parser.add_argument("--headless", action="store_true", help="Использовать headless для Playwright.")
    parser.add_argument("--export-format", choices=("csv", "json"), default="csv")
    parser.add_argument("--export-dir", default=".")
    args = parser.parse_args()

    init_db()

    if HAVE_PLAYWRIGHT:
        logging.info("Playwright доступен — будет использован для рендеринга (если выбран).")
    elif HAVE_REQS_BS:
        logging.info("Playwright недоступен — используется fallback: requests + BeautifulSoup (без JS).")
    else:
        logging.error("Ни Playwright, ни requests+BeautifulSoup не установлены. Установите пакеты:")
        logging.error("  pip install playwright && playwright install")
        logging.error("  pip install requests bs4")
        sys.exit(1)

    logging.info("Запуск: query=%s max_items=%s max_pages=%s", args.query, args.max_items, args.max_pages)
    items = fetch_search_results(args.query, max_items=args.max_items, max_pages=args.max_pages, headless=args.headless)
    ts = datetime.now(timezone.utc).isoformat()  # исправлено: timezone-aware UTC
    saved = 0
    for prod in items:
        save_product_to_db(prod, ts, args.query)
        saved += 1
    logging.info("Сохранено %d товаров в %s", saved, DB_PATH)
    export_data(db_path=DB_PATH, export_dir=args.export_dir, fmt=args.export_format)
    logging.info("Готово.")


if __name__ == "__main__":
    main()
