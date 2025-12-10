import streamlit as st
import pandas as pd
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib.robotparser
from collections import defaultdict

# -------------------------
# Ethical scraping utilities
# -------------------------

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]

# cache robot parsers per origin to avoid repeated downloads
_robot_parsers = {}

def get_robot_parser(origin):
    if origin in _robot_parsers:
        return _robot_parsers[origin]
    robots_url = urljoin(origin, "/robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        # If robots.txt cannot be fetched, default to cautious: treat as disallow unknown aggressive crawling
        rp = None
    _robot_parsers[origin] = rp
    return rp

def is_allowed_by_robots(url, user_agent="*"):
    origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    rp = get_robot_parser(origin)
    if rp is None:
        # when robots cannot be determined, be conservative and allow but with strict rate limiting shown to user
        return True
    return rp.can_fetch(user_agent, url)

def get_session():
    s = requests.Session()
    s.headers.update({
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    })
    return s

def fetch_with_retries(url, session, retries=3, timeout=15, backoff_factor=1.5):
    """
    Polite fetch: rotates user agent, respects simple backoff on 429/5xx, returns text or None.
    This is intended for ethical scraping only (use with permission / respect robots.txt).
    """
    for attempt in range(1, retries + 1):
        session.headers["User-Agent"] = random.choice(USER_AGENTS)
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 503) or 500 <= r.status_code < 600:
                sleep_for = backoff_factor ** attempt + random.uniform(0, 1)
                time.sleep(sleep_for)
                continue
            # other non-success statuses -> stop trying
            return None
        except requests.RequestException:
            sleep_for = backoff_factor ** attempt + random.uniform(0, 1)
            time.sleep(sleep_for)
    return None

# -------------------------
# Parser (ethical mode)
# -------------------------

class EthicalParser:
    def __init__(self, catalog_url, max_pages=3, min_delay=2.0, max_delay=5.0):
        self.catalog_url = catalog_url
        self.max_pages = max_pages
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.session = get_session()
        self.product_links = []
        self.data = []
        self.domain_crawl_times = defaultdict(lambda: 0.0)  # last request time per domain

    def wait_if_needed(self, url):
        """
        Enforce a polite per-domain delay so we don't hammer the server.
        """
        domain = urlparse(url).netloc
        now = time.time()
        min_interval = self.min_delay
        elapsed = now - self.domain_crawl_times[domain]
        if elapsed < min_interval:
            to_sleep = min_interval - elapsed + random.uniform(0, 0.5)
            time.sleep(to_sleep)
        self.domain_crawl_times[domain] = time.time()

    def get_page_links(self, page_url):
        # Check robots.txt
        if not is_allowed_by_robots(page_url):
            st.warning(f"Согласно robots.txt доступ к {page_url} запрещён или не рекомендуется.")
            return []

        self.wait_if_needed(page_url)
        html = fetch_with_retries(page_url, self.session)
        if not html:
            st.error(f"Не удалось загрузить страницу: {page_url}")
            return []
        soup = BeautifulSoup(html, "html.parser")
        # Generic safe selector: try to find product links by common patterns
        anchors = soup.find_all("a", href=True)
        hrefs = set()
        for a in anchors:
            href = a["href"]
            # normalize relative urls
            href = urljoin(page_url, href)
            # simple heuristic: product pages often contain '/product', '/item', '/p/' or a long id
            if any(token in href for token in ["/product", "/item", "/p/", "/good/"]) or len(urlparse(href).path.split("/")) > 3:
                hrefs.add(href.split("?")[0])
        return list(hrefs)

    def get_product_details(self, url):
        if not is_allowed_by_robots(url):
            st.info(f"Ресурс {url} не позволяет парсить (robots.txt). Пропускаем.")
            return
        self.wait_if_needed(url)
        html = fetch_with_retries(url, self.session)
        if not html:
            st.error(f"Не удалось получить товар: {url}")
            return
        soup = BeautifulSoup(html, "html.parser")
        # Try several common tags for title/price/image
        title = None
        price = None
        image = None

        # title candidates
        for sel in ["h1", "[data-testid~=title]", ".product-title", ".title", "meta[property='og:title']"]:
            try:
                if sel.startswith("meta"):
                    m = soup.select_one(sel)
                    if m and m.get("content"):
                        title = m["content"]
                        break
                else:
                    el = soup.select_one(sel)
                    if el and el.get_text(strip=True):
                        title = el.get_text(strip=True)
                        break
            except Exception:
                continue

        # price candidates
        for sel in [".price", "[data-testid~=price]", ".product-price", "meta[property='product:price:amount']"]:
            try:
                if sel.startswith("meta"):
                    m = soup.select_one(sel)
                    if m and m.get("content"):
                        price = m["content"]
                        break
                else:
                    el = soup.select_one(sel)
                    if el and el.get_text(strip=True):
                        price = el.get_text(strip=True)
                        break
            except Exception:
                continue

        # image candidates
        for sel in ["meta[property='og:image']", "img", "[data-testid~=image]"]:
            try:
                if sel.startswith("meta"):
                    m = soup.select_one(sel)
                    if m and m.get("content"):
                        image = m["content"]
                        break
                else:
                    el = soup.select_one(sel)
                    if el and el.get("src"):
                        image = urljoin(url, el["src"])
                        break
            except Exception:
                continue

        self.data.append({
            "Название": title or "Нет названия",
            "Цена": price or "Нет цены",
            "Ссылка": url,
            "Фото": image or "",
        })

    def run(self):
        # Iterate pages
        for page in range(1, self.max_pages + 1):
            page_url = self.catalog_url.replace("{page}", str(page))
            st.write(f"Загрузка страницы: {page_url}")
            links = self.get_page_links(page_url)
            st.write(f"Найдено ссылок (потенциальных товаров): {len(links)}")
            self.product_links.extend(links)
            time.sleep(random.uniform(self.min_delay, self.max_delay))

        st.write(f"Всего кандидатов: {len(self.product_links)}")
        # Limit total items to something reasonable
        for idx, url in enumerate(self.product_links, 1):
            st.write(f"[{idx}/{len(self.product_links)}] Обработка: {url}")
            self.get_product_details(url)
            time.sleep(random.uniform(self.min_delay, self.max_delay))
        return self.data

# -------------------------
# Streamlit UI
# -------------------------

def main():
    st.title("Этичный парсер маркетплейсов (без обхода защит)")
    st.write("Этот инструмент демонстрирует аккуратный подход к сбору данных: проверка robots.txt, ограничение частоты запросов и экспоненциальные повторы.")
    catalog_url = st.text_input(
        "URL каталога (используйте {page} для номера страницы)",
        value="https://example.com/catalog?page={page}"
    )
    max_pages = st.number_input("Максимальное число страниц", min_value=1, max_value=50, value=2)
    min_delay = st.number_input("Минимальная задержка между запросами (сек)", min_value=0.5, value=2.0, step=0.5)
    max_delay = st.number_input("Максимальная задержка между запросами (сек)", min_value=0.5, value=4.0, step=0.5)

    if st.button("Запустить (только этический режим)"):
        parser = EthicalParser(catalog_url, max_pages=int(max_pages), min_delay=float(min_delay), max_delay=float(max_delay))
        with st.spinner("Идёт парсинг (пожалуйста, убедитесь, что у вас есть право парсить этот сайт)..."):
            data = parser.run()
        df = pd.DataFrame(data)
        st.success("Готово")
        st.dataframe(df)
        if not df.empty:
            df.to_excel("data.xlsx", index=False)
            with open("data.xlsx", "rb") as f:
                st.download_button("Скачать Excel", f, "data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
