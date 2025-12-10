import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser
import time
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import io
from requests.adapters import HTTPAdapter, Retry

# ------- Настройки и константы -------
# Набор User-Agent'ов для ротации
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
    " Chrome/116.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)"
    " Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
    " Chrome/115.0 Safari/537.36",
]
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}

# Selectors (оставлены ваши, можно донастроить)
SELECTORS = {
    "ozon": {
        "card": ("div", {"class": "b5v1"}),
        "title": ("a", {"class": "a4d3"}),
        "price": ("div", {"class": "b5v2"}),
        "img": ("img", {}),
        "base_domain": "https://ozon.ru",
    },
    "wildberries": {
        "card": ("div", {"class": "product-card"}),
        "title": ("a", {"class": "product-card__name"}),
        "price_alt": [("ins", {"class": "price__new"}), ("ins", {"class": "price__old"})],
        "img": ("img", {"class": "product-card__image"}),
        "base_domain": "https://www.wildberries.ru",
    },
}

# Политика повторов
RETRY_STRATEGY = Retry(
    total=3,
    status_forcelist=(429, 500, 502, 503, 504),
    backoff_factor=1,
    allowed_methods=frozenset(["GET", "POST"]),
)

ROBOTS_CACHE = {}
ROBOTS_LOCK = threading.Lock()
DOMAIN_RATE = {}
DOMAIN_LOCK = threading.Lock()
THREAD_LOCAL = threading.local()

# ------- Вспомогательные функции -------

def make_session():
    """Создает requests.Session с retry и случайным User-Agent. Используется в потоках."""
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    s.headers["User-Agent"] = random.choice(USER_AGENTS)
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def get_thread_session():
    """Возвращает сессию для текущего потока (thread-local)."""
    if not getattr(THREAD_LOCAL, "session", None):
        THREAD_LOCAL.session = make_session()
    return THREAD_LOCAL.session

def get_robot_parser_for(url):
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    with ROBOTS_LOCK:
        if base in ROBOTS_CACHE:
            return ROBOTS_CACHE[base]
        rp = robotparser.RobotFileParser()
        try:
            rp.set_url(urljoin(base, "/robots.txt"))
            rp.read()
            ROBOTS_CACHE[base] = rp
        except Exception:
            # Если robots.txt недоступен, кешируем None, но не запрещаем автоматично
            ROBOTS_CACHE[base] = None
        return ROBOTS_CACHE[base]

def is_allowed(url):
    rp = get_robot_parser_for(url)
    if rp is None:
        # robots.txt не прочитан — осторожно разрешаем (с замедлением), но логируем
        return True
    # используем базовый User-Agent (не полный случайный)
    ua = USER_AGENTS[0]
    try:
        return rp.can_fetch(ua, url)
    except Exception:
        return True

def wait_for_domain(host, min_interval):
    """Гарантирует паузу между запросами к одному домену."""
    with DOMAIN_LOCK:
        last = DOMAIN_RATE.get(host, 0)
        now = time.time()
        wait = max(0, min_interval - (now - last))
        if wait > 0:
            time.sleep(wait)
        DOMAIN_RATE[host] = time.time()

def safe_get(url, delay_between_requests=1.0, timeout=15):
    """Безопасный GET: rate limit по домену, проверка robots, детекция капчи, retries handled by session adapter."""
    if not is_allowed(url):
        return None, {"error": "robots.txt запрещает доступ"}
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    # Немного случайности (чтобы имитировать поведение человека)
    jitter = random.uniform(0.2, 0.6)
    wait_for_domain(host, delay_between_requests + jitter)
    session = get_thread_session()
    # Периодически менять User-Agent у сессии (чтобы не выглядеть подозрительно)
    if random.random() < 0.05:
        session.headers["User-Agent"] = random.choice(USER_AGENTS)
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        text = resp.text
        # Простая детекция блокировок/капч
        lower = text[:5000].lower()
        if "captcha" in lower or "are you a human" in lower or "access denied" in lower:
            return resp, {"error": "blocked_or_captcha_detected"}
        return resp, None
    except requests.exceptions.RequestException as e:
        return None, {"error": str(e)}

# ------- Извлечение данных -------
def parse_json_ld(soup):
    data = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            j = json.loads(script.string or "{}")
            if isinstance(j, dict):
                if j.get("@type", "").lower() == "product" or "name" in j:
                    data.update(j)
            elif isinstance(j, list):
                for elem in j:
                    if isinstance(elem, dict) and (elem.get("@type", "").lower() == "product" or "name" in elem):
                        data.update(elem)
        except Exception:
            continue
    return data

def extract_attributes_from_detail_soup(soup):
    attrs = {}
    attrs.update({f"ld_{k}": v for k, v in parse_json_ld(soup).items()})
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                k = cols[0].get_text(strip=True)
                v = cols[1].get_text(strip=True)
                if k:
                    attrs.setdefault(k, v)
    for dl in soup.find_all("dl"):
        dt_tags = dl.find_all("dt")
        dd_tags = dl.find_all("dd")
        for a, b in zip(dt_tags, dd_tags):
            k = a.get_text(strip=True)
            v = b.get_text(strip=True)
            if k:
                attrs.setdefault(k, v)
    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True)
        if ":" in text:
            try:
                k, v = map(str.strip, text.split(":", 1))
                if k:
                    attrs.setdefault(k, v)
            except Exception:
                continue
    for name in ("description", "keywords"):
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            attrs.setdefault(name, meta["content"])
    return attrs

def extract_card_info(card, conf):
    title_tag = card.find(*conf["title"])
    title = title_tag.get_text(strip=True) if title_tag else "Нет названия"
    link = "Нет ссылки"
    if title_tag and title_tag.has_attr("href"):
        link = urljoin(conf.get("base_domain", ""), title_tag["href"])
    price = "Нет цены"
    if "price" in conf:
        p = card.find(*conf["price"])
        if p:
            price = p.get_text(strip=True)
    elif "price_alt" in conf:
        for t, a in conf["price_alt"]:
            p = card.find(t, attrs=a)
            if p and p.get_text(strip=True):
                price = p.get_text(strip=True)
                break
    img = card.find(*conf["img"])
    img_url = "Нет фото"
    if img and img.has_attr("src"):
        img_url = img["src"]
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif img_url.startswith("/"):
            img_url = urljoin(conf.get("base_domain", ""), img_url)
    return {"Название": title, "Ссылка": link, "Цена": price, "Фото": img_url}

def fetch_product_detail(url, delay_between_requests=1.0):
    if not url or url in ("Нет ссылки", None):
        return {}
    resp, err = safe_get(url, delay_between_requests=delay_between_requests)
    if err:
        return err
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        detail = {}
        h1 = soup.find("h1")
        if h1:
            detail["detail_title"] = h1.get_text(strip=True)
        detail.update(extract_attributes_from_detail_soup(soup))
        return detail
    except Exception as e:
        return {"error": f"parse_error: {e}"}

def process_product(base_info, delay_details):
    if base_info.get("Ссылка") not in ("Нет ссылки", None):
        details = fetch_product_detail(base_info["Ссылка"], delay_between_requests=delay_details)
    else:
        details = {}
    out = {**base_info}
    if isinstance(details, dict):
        out.update(details)
    else:
        out["detail_error"] = str(details)
    return out

# ------- Основной скрапер -------
def scrape_seller(seller_url, site_key, max_pages=5, delay_pages=1.0, delay_details=1.0, max_workers=5, progress_callback=None):
    conf = SELECTORS.get(site_key)
    if not conf:
        raise ValueError("Сайт не настроен в SELECTORS")
    # Проверка robots для базового URL (но не блокируем, если robots недоступен)
    if not is_allowed(seller_url):
        if progress_callback:
            progress_callback(f"robots.txt явно запрещает доступ к {seller_url}. Остановка.")
        return []

    products = []
    for page in range(1, max_pages + 1):
        page_url = f"{seller_url.rstrip('/')}" + f"?page={page}"
        if not is_allowed(page_url):
            if progress_callback:
                progress_callback(f"robots.txt запрещает доступ к {page_url}. Остановка.")
            break
        resp, err = safe_get(page_url, delay_between_requests=delay_pages)
        if err:
            if progress_callback:
                progress_callback(f"Ошибка загрузки страницы {page_url}: {err.get('error')}")
            break
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            if progress_callback:
                progress_callback(f"Не удалось распарсить страницу {page}: {e}")
            break
        tag, attrs = conf["card"]
        cards = soup.find_all(tag, attrs=attrs)
        if not cards:
            if progress_callback:
                progress_callback(f"На странице {page} карточки не найдены — предполагаем конец.")
            break
        if progress_callback:
            progress_callback(f"Обработка страницы {page}, карточек: {len(cards)}")
        # Обработка карточек в потоках; используем thread-local сессии внутри safe_get
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for card in cards:
                base = extract_card_info(card, conf)
                futures.append(executor.submit(process_product, base, delay_details))
            for f in as_completed(futures):
                try:
                    products.append(f.result())
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Ошибка при обработке товара: {e}")
        # Пауза между страницами с небольшой случайностью
        time.sleep(delay_pages + random.uniform(0.1, 0.7))
    return products

def detect_site_from_url(url):
    dom = urlparse(url).netloc.lower()
    if "ozon." in dom:
        return "ozon"
    if "wildberries" in dom:
        return "wildberries"
    return None

# ------- Streamlit UI -------
def main():
    st.title("Улучшенный парсер OZON и Wildberries (с мерами против блокировок)")
    url = st.text_input("Введите URL магазина/продавца")
    max_pages = st.number_input("Максимум страниц", min_value=1, max_value=200, value=5)
    delay_pages = st.number_input("Задержка между страницами (сек)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    delay_details = st.number_input("Задержка между запросами деталей (сек)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    max_workers = st.number_input("Потоков обработки", min_value=1, max_value=20, value=4)

    if st.button("Начать парсинг"):
        if not url:
            st.warning("Пожалуйста, введите URL.")
            return

        site_key = detect_site_from_url(url)
        if not site_key:
            s = st.text_input("Не удалось определить сайт. Введите 'ozon' или 'wildberries':")
            if not s:
                st.error("Введите сайт.")
                return
            site_key = s.strip().lower()
            if site_key not in ("ozon", "wildberries"):
                st.error("Некорректный ввод сайта.")
                return

        progress_text = st.empty()
        df_container = st.empty()

        def progress_callback(msg):
            try:
                progress_text.text(msg)
            except Exception:
                pass

        with st.spinner("Парсинг запущен... (будьте терпеливы, скрипт делает паузы чтобы снизить риск блокировки)"):
            items = scrape_seller(
                url,
                site_key,
                max_pages=int(max_pages),
                delay_pages=float(delay_pages),
                delay_details=float(delay_details),
                max_workers=int(max_workers),
                progress_callback=progress_callback,
            )

        if not items:
            st.info("Ничего не найдено или доступ запрещен/ошибка.")
            return

        df = pd.DataFrame(items)
        st.success(f"Найдено {len(df)} товаров.")
        st.dataframe(df)

        # Подготовка Excel в памяти и безопасная отдача
        buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="products")
            buffer.seek(0)
            st.download_button(
                label="Скачать Excel",
                data=buffer.getvalue(),
                file_name=f"{site_key}_products.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Не удалось сформировать файл Excel: {e}")

if __name__ == "__main__":
    main()
