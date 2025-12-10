# streamlit_scraper_full_commented.py
"""
Streamlit приложение: устойчивый скрейпер продавцов OZON / Wildberries с комментариями на русском.

Использование:
  streamlit run streamlit_scraper_full_commented.py

Особенности:
- Простая проверка зависимостей.
- Сессия requests с автоматическими повторными попытками.
- Ограничение скорости и jitter для вежливого скрейпинга.
- Проверка robots.txt (по возможности).
- Извлечение карточек товаров (название, ссылка, цена, изображение) и загрузка деталей товара.
- Параллельная загрузка деталей товаров.
- Выгрузка результатов в XLSX или JSON.
- Обработка ошибок и логирование прогресса.

Примечание:
- CSS-селекторы — пример, может потребоваться обновление под текущий HTML сайта.
- Уважайте правила сайтов и robots.txt.
"""

from __future__ import annotations
import time
import random
import threading
import json
import io
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse

# ---- Проверка зависимостей ----
try:
    import streamlit as st
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from urllib.robotparser import RobotFileParser
except Exception as e:
    msg = f"Отсутствует зависимость (запустите вне Streamlit или установите): {e}"
    try:
        print(msg)
    except Exception:
        pass
    raise

# ---- Константы и настройки ----
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, как Gecko) Chrome/116.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, как Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, как Gecko) Chrome/115.0 Safari/537.36",
]
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}
# Примеры селекторов; актуальные селекторы могут отличаться
SELECTORS = {
    "ozon": {
        "card": ("div", {"class": "b5v1"}),
        "title": ("a", {"class": "a4d3"}),
        "price": ("div", {"class": "b5v2"}),
        "img": ("img", {}),
        "base_domain": "https://ozon.ru",
        "page_param": "?page={page}",
    },
    "wildberries": {
        "card": ("div", {"class": "product-card"}),
        "title": ("a", {"class": "product-card__name"}),
        "price_alt": [("ins", {"class": "price__new"}), ("ins", {"class": "price__old"})],
        "img": ("img", {"class": "product-card__image"}),
        "base_domain": "https://www.wildberries.ru",
        "page_param": "?page={page}",
    },
}
# Настройки для повторных попыток запросов
RETRY_STRATEGY = Retry(total=3, status_forcelist=(429, 500, 502, 503, 504),
                       backoff_factor=1, allowed_methods=frozenset({"GET"}))
# Текущая сессия в потоке
THREAD_LOCAL = threading.local()
# Кэш robots.txt для доменов
ROBOTS_CACHE: Dict[str, Optional[RobotFileParser]] = {}
ROBOTS_LOCK = threading.Lock()
# Ограничение скорости по доменам
DOMAIN_RATE: Dict[str, float] = {}
DOMAIN_LOCK = threading.Lock()

# ---- Вспомогательные функции для сети ----
def make_session() -> requests.Session:
    """Создать сессию requests с повторными попытками и настроенными заголовками."""
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    s.headers["User-Agent"] = random.choice(USER_AGENTS)
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter); s.mount("https://", adapter)
    return s

def get_thread_session() -> requests.Session:
    """Вернуть сессию для текущего потока."""
    if not getattr(THREAD_LOCAL, "session", None):
        THREAD_LOCAL.session = make_session()
    return THREAD_LOCAL.session

def get_robot_parser(base_url: str) -> Optional[RobotFileParser]:
    """Получить или скачать robots.txt для базового URL."""
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    with ROBOTS_LOCK:
        if base in ROBOTS_CACHE:
            return ROBOTS_CACHE[base]
    rp = RobotFileParser()
    robots_url = urljoin(base, "/robots.txt")
    try:
        rp.set_url(robots_url)
        rp.read()
        with ROBOTS_LOCK:
            ROBOTS_CACHE[base] = rp
        return rp
    except Exception:
        with ROBOTS_LOCK:
            ROBOTS_CACHE[base] = None
        return None

def allowed_by_robots(url: str, user_agent: str = "*") -> bool:
    """Проверить разрешение robots.txt (по возможности)."""
    rp = get_robot_parser(url)
    if rp is None:
        return True  # не удалось прочитать robots.txt, считаем разрешение
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True

def wait_for_domain(host: str, min_interval: float) -> None:
    """Обеспечить задержку между запросами к одному домену."""
    with DOMAIN_LOCK:
        last = DOMAIN_RATE.get(host, 0.0)
        now = time.time()
        to_wait = max(0.0, min_interval - (now - last))
        if to_wait:
            time.sleep(to_wait)
        DOMAIN_RATE[host] = time.time()

def safe_get(url: str, delay_between_requests: float = 1.0, timeout: int = 15,
             ignore_robots: bool = False) -> Tuple[Optional[requests.Response], Optional[Dict[str, str]]]:
    """
    Выполнить GET-запрос с учетом robots.txt, ограничения скорости и случайным jitter.
    Возвращает (ответ, None) при успехе или (ответ или None, словарь ошибки) при ошибке.
    """
    if not ignore_robots and not allowed_by_robots(url):
        return None, {"error": "robots_disallow"}
    host = urlparse(url).netloc.lower()
    # Жадность: ждём необходимое время для ограничения скорости
    wait_for_domain(host, delay_between_requests + random.uniform(0.2, 0.6))
    session = get_thread_session()
    # Иногда меняем User-Agent для разнообразия
    if random.random() < 0.05:
        session.headers["User-Agent"] = random.choice(USER_AGENTS)
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        # Проверка на капчу или блокировку
        snippet = (r.text or "")[:5000].lower()
        if any(token in snippet for token in ("captcha", "are you a human", "access denied", "verify you are human")):
            return r, {"error": "blocked_or_captcha"}
        return r, None
    except requests.RequestException as e:
        return None, {"error": str(e)}

# ---- Вспомогательные функции для парсинга ----
def parse_json_ld(soup: BeautifulSoup) -> Dict[str, Any]:
    """Извлечь JSON-LD данные из скриптов."""
    out: Dict[str, Any] = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            j = json.loads(script.string or "{}")
            if isinstance(j, dict):
                out.update(j)
            elif isinstance(j, list):
                for e in j:
                    if isinstance(e, dict):
                        out.update(e)
        except Exception:
            continue
    return out

def extract_attributes_from_detail_soup(soup: BeautifulSoup) -> Dict[str, Any]:
    """Попытка извлечь атрибуты из страницы товара."""
    attrs: Dict[str, Any] = {}
    attrs.update({f"ld_{k}": v for k, v in parse_json_ld(soup).items()})
    # Таблицы
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                k = cols[0].get_text(strip=True); v = cols[1].get_text(strip=True)
                if k: attrs.setdefault(k, v)
    # Определённые списки
    for dl in soup.find_all("dl"):
        dt = dl.find_all("dt"); dd = dl.find_all("dd")
        for a, b in zip(dt, dd):
            k = a.get_text(strip=True); v = b.get_text(strip=True)
            if k: attrs.setdefault(k, v)
    # Листинги с двоеточием
    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True)
        if ":" in text:
            try:
                k, v = map(str.strip, text.split(":", 1))
                if k: attrs.setdefault(k, v)
            except Exception:
                pass
    # Мета-теги
    for name in ("description", "keywords"):
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            attrs.setdefault(name, meta["content"])
    return attrs

def extract_card_info(card: BeautifulSoup, conf: Dict[str, Any]) -> Dict[str, str]:
    """Извлечь информацию о товаре из карточки по селекторам."""
    title_tag = card.find(*conf["title"])
    title = title_tag.get_text(strip=True) if title_tag else "Нет названия"
    link = "Нет ссылки"
    if title_tag and title_tag.has_attr("href"):
        link = urljoin(conf.get("base_domain", ""), title_tag["href"])
    price = "Нет цены"
    if "price" in conf:
        p = card.find(*conf["price"])
        if p: price = p.get_text(strip=True)
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
        if img_url.startswith("//"): img_url = "https:" + img_url
        elif img_url.startswith("/"): img_url = urljoin(conf.get("base_domain", ""), img_url)
    return {"Название": title, "Ссылка": link, "Цена": price, "Фото": img_url}

def fetch_product_detail(url: str, delay_between_requests: float = 1.0,
                         ignore_robots: bool = False) -> Dict[str, Any]:
    """Загрузить страницу товара и извлечь атрибуты."""
    if not url or url in ("Нет ссылки", None):
        return {}
    resp, err = safe_get(url, delay_between_requests=delay_between_requests, ignore_robots=ignore_robots)
    if err:
        return err
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        detail: Dict[str, Any] = {}
        h1 = soup.find("h1")
        if h1:
            detail["detail_title"] = h1.get_text(strip=True)
        detail.update(extract_attributes_from_detail_soup(soup))
        return detail
    except Exception as e:
        return {"error": f"parse_error: {e}"}

def process_product(base_info: Dict[str, str], delay_details: float,
                    ignore_robots: bool = False) -> Dict[str, Any]:
    """Параллельное получение деталей товара и объединение данных."""
    if base_info.get("Ссылка") not in ("Нет ссылки", None):
        details = fetch_product_detail(base_info["Ссылка"], delay_between_requests=delay_details, ignore_robots=ignore_robots)
    else:
        details = {}
    out = dict(base_info)
    if isinstance(details, dict):
        out.update(details)
    else:
        out["detail_error"] = str(details)
    return out

# ---- Основные функции скрейпинга ----
def detect_site_from_url(url: str) -> Optional[str]:
    """Определить сайт по URL."""
    dom = urlparse(url).netloc.lower()
    if "ozon." in dom:
        return "ozon"
    if "wildberries" in dom or "wbx" in dom:
        return "wildberries"
    return None

def scrape_seller(seller_url: str,
                  site_key: str,
                  max_pages: int = 3,
                  delay_pages: float = 1.0,
                  delay_details: float = 1.0,
                  max_workers: int = 4,
                  progress_callback: Optional[callable] = None,
                  ignore_robots: bool = False) -> List[Dict[str, Any]]:
    """
    Перебор страниц, парсинг карточек и параллельная загрузка деталей.
    Возвращает список товаров.
    """
    conf = SELECTORS.get(site_key)
    if not conf:
        raise ValueError("Конфигурация для сайта отсутствует")
    items: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        page_suffix = conf.get("page_param", "?page={page}").format(page=page)
        page_url = f"{seller_url.rstrip('/')}{page_suffix}"
        if progress_callback: progress_callback(f"Загрузка {page_url}")
        # Используем функцию с управлением задержками
        resp, err = safe_get(page_url, delay_between_requests=delay_pages, ignore_robots=ignore_robots)
        if err:
            if progress_callback: progress_callback(f"Ошибка страницы: {err.get('error')}")
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        tag, attrs = conf["card"]
        cards = soup.find_all(tag, attrs=attrs)
        if not cards:
            if progress_callback: progress_callback("Карточки не найдены, остановка пагинации.")
            break
        if progress_callback: progress_callback(f"Найдено {len(cards)} карточек на странице {page}")
        # Параллельная загрузка деталей
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_product, extract_card_info(c, conf), delay_details, ignore_robots) for c in cards]
            for f in as_completed(futures):
                try:
                    res = f.result()
                    items.append(res)
                except Exception as e:
                    if progress_callback: progress_callback(f"Ошибка товара: {e}")
        # Небольшая задержка между страницами
        time.sleep(delay_pages + random.uniform(0.05, 0.4))
    return items

# ---- UI Streamlit ----
st.set_page_config(page_title="OZON/Wildberries Scraper", layout="wide")
st.title("OZON / Wildberries Scraper (Streamlit)")

col1, col2 = st.columns([2, 1])
with col1:
    seller_url = st.text_input("URL продавца/магазина")
    site_choice = st.selectbox("Сайт", options=["auto", "ozon", "wildberries"], index=0)
    max_pages = st.number_input("Макс. страниц", min_value=1, max_value=200, value=3)
    delay_pages = st.number_input("Задержка между страницами (с)", min_value=0.0, value=1.0, step=0.1)
    delay_details = st.number_input("Задержка между деталями (с)", min_value=0.0, value=1.0, step=0.1)
    workers = st.number_input("Потоки", min_value=1, max_value=20, value=4)
    ignore_robots = st.checkbox("Игнорировать robots.txt (не рекомендуется)", value=False)

with col2:
    out_name = st.text_input("Имя файла для выгрузки", value="products.xlsx")
    start_btn = st.button("Начать скрейпинг")
    st.markdown("Примечания:")
    st.markdown("- Селекторы — пример, под текущий сайт нужно адаптировать.")
    st.markdown("- Используйте небольшое количество страниц и задержки, чтобы избежать блокировок.")

log_box = st.empty()
result_box = st.empty()

def append_log(msg: str) -> None:
    """Добавлять сообщение в лог и отображать."""
    if "log" not in st.session_state:
        st.session_state["log"] = []
    st.session_state["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    log_box.text_area("Лог прогресса", value="\n".join(st.session_state["log"]), height=240)

if start_btn:
    st.session_state["log"] = []
    if not seller_url:
        st.error("Пожалуйста, введите URL продавца/магазина")
    else:
        # Автоматическое определение сайта
        site = None if site_choice == "auto" else site_choice
        if site is None:
            site = detect_site_from_url(seller_url)
        if not site:
            st.error("Не удалось определить сайт по URL. Выберите 'ozon' или 'wildberries'.")
        else:
            append_log(f"Запуск скрейпа для сайта='{site}', url='{seller_url}'")
            with st.spinner("Идет скрейпинг..."):
                try:
                    # Основной вызов функции скрейпинга
                    items = scrape_seller(
                        seller_url=seller_url,
                        site_key=site,
                        max_pages=int(max_pages),
                        delay_pages=float(delay_pages),
                        delay_details=float(delay_details),
                        max_workers=int(workers),
                        progress_callback=append_log,
                        ignore_robots=ignore_robots,
                    )
                except Exception as e:
                    append_log(f"Фатальная ошибка: {e}")
                    st.exception(e)
                    items = []
            if not items:
                append_log("Нет найденных товаров или ошибка/блокировка.")
                st.info("Нет товаров или возникла ошибка. Посмотрите лог.")
            else:
                df = pd.DataFrame(items)
                append_log(f"Получено {len(df)} товаров; формируем файл.")
                result_box.dataframe(df)
                buf = io.BytesIO()
                try:
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False, sheet_name="products")
                    buf.seek(0)
                    st.download_button("Скачать Excel", data=buf.getvalue(), file_name=out_name,
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except:
                    # В случае ошибки — скачать JSON
                    st.download_button("Скачать JSON", data=json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8"),
                                       file_name=out_name.rsplit(".", 1)[0] + ".json", mime="application/json")
                append_log("Завершено. Можно скачать результаты.")
