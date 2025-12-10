#!/usr/bin/env python3
"""
full_scraper_with_installer.py

Полный самодостаточный скрипт парсера (OZON / Wildberries) с автоматической
попыткой установки зависимостей при старте, включая создание виртуального окружения.
"""

import sys
import os
import venv
import subprocess
import time
import random
import threading
import json
import io
import logging
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse

# ================== Встроенная логика автозапуска в виртуальном окружении ==================

def in_virtualenv():
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def create_virtualenv(env_dir: str = "venv") -> bool:
    if os.path.exists(env_dir):
        print(f"[info] Виртуальное окружение уже существует по пути '{env_dir}'.")
        return True
    try:
        print(f"[info] Создаю виртуальное окружение в '{env_dir}'...")
        venv.create(env_dir, with_pip=True)
        print("[info] Виртуальное окружение создано.")
        return True
    except Exception as e:
        print(f"[error] Не удалось создать виртуальное окружение: {e}")
        return False

def restart_in_venv(env_dir: str = "venv"):
    if sys.platform == "win32":
        python_executable = os.path.join(env_dir, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(env_dir, "bin", "python")
    if not os.path.exists(python_executable):
        print(f"[error] Python-исполнитель не найден по пути {python_executable}")
        sys.exit(1)
    print(f"[info] Перезапуск внутри виртуального окружения...")
    os.execv(python_executable, [python_executable] + sys.argv)

# Проверка, запущен ли уже внутри виртуального окружения
if not in_virtualenv():
    env_dir = "venv"
    if create_virtualenv(env_dir):
        restart_in_venv(env_dir)
    else:
        print("[error] Не удалось создать виртуальное окружение. Установите зависимости вручную.")
        sys.exit(1)

# ================== Далее идет ваш скрипт ==================

# Ваш исходный код начинается здесь...

# from __future__ import annotations
import sys
import os
import subprocess
import time
import random
import threading
import json
import io
import logging
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse

# ----------------- Automatic dependency installer -----------------
REQUIREMENTS = {
    "requests": "requests",
    "bs4": "beautifulsoup4",
    "pandas": "pandas",
    "xlsxwriter": "XlsxWriter",
    "urllib3": "urllib3",
}
OPTIONAL = {
    "streamlit": "streamlit",
}
INSTALL_LOG = "install_log.txt"

def try_install_missing():
    missing = []
    for mod, pkg in REQUIREMENTS.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    cmd = [sys.executable, "-m", "pip", "install"] + missing
    print("[info] Обнаружены отсутствующие пакеты: ", ", ".join(missing))
    print(f"[info] Пытаюсь установить: {' '.join(missing)} (лог -> {INSTALL_LOG})")
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600, check=True)
        out = proc.stdout or b""
        with open(INSTALL_LOG, "wb") as f:
            f.write(out)
        print(f"[info] pip завершил работу (код={proc.returncode}), смотрите {INSTALL_LOG}")
    except subprocess.TimeoutExpired:
        with open(INSTALL_LOG, "a", encoding="utf-8") as f:
            f.write("\n[pip install timed out]\n")
        print("[error] Время выполнения pip превысило лимит, смотрите", INSTALL_LOG)
    except subprocess.CalledProcessError as e:
        with open(INSTALL_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[pip install error] {e}\n")
        print("[error] Ошибка при выполнении pip, смотрите", INSTALL_LOG)

try_install_missing()

# Try imports; if still missing, inform user and exit
try:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from concurrent.futures import ThreadPoolExecutor, as_completed
except Exception:
    print("[fatal] Не удалось импортировать необходимые библиотеки после попытки установки.")
    print("Проверьте", INSTALL_LOG, "и установите вручную:")
    pkgs = list(REQUIREMENTS.values()) + list(OPTIONAL.values())
    print("  python -m pip install " + " ".join(pkgs))
    sys.exit(1)

# Optional GUI
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# ----------------- Логирование -----------------
LOG_FILE = "scraper.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOG_FILE, encoding="utf-8")])
logger = logging.getLogger(__name__)

# ----------------- Конфигурация -----------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
]
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}
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
RETRY_STRATEGY = Retry(total=3, status_forcelist=(429, 500, 502, 503, 504), backoff_factor=1, allowed_methods=frozenset({"GET"}))
ROBOTS_CACHE: Dict[str, Optional[object]] = {}
ROBOTS_LOCK = threading.Lock()
DOMAIN_RATE: Dict[str, float] = {}
DOMAIN_LOCK = threading.Lock()
THREAD_LOCAL = threading.local()

# ----------------- Вспомогательные функции сети -----------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    s.headers["User-Agent"] = random.choice(USER_AGENTS)
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def get_thread_session() -> requests.Session:
    if not getattr(THREAD_LOCAL, "session", None):
        THREAD_LOCAL.session = make_session()
    return THREAD_LOCAL.session

def get_robots_text(base_url: str) -> Optional[str]:
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    with ROBOTS_LOCK:
        if base in ROBOTS_CACHE:
            return ROBOTS_CACHE[base]
    try:
        r = requests.get(urljoin(base, "/robots.txt"), headers={"User-Agent": USER_AGENTS[0]}, timeout=6)
        text = r.text if r.status_code == 200 else None
    except Exception:
        text = None
    with ROBOTS_LOCK:
        ROBOTS_CACHE[base] = text
    return text

def is_allowed(url: str) -> bool:
    txt = get_robots_text(url)
    if not txt:
        return True
    lines = [l.strip() for l in txt.splitlines() if l.strip() and not l.strip().startswith("#")]
    ua = None
    disallows = []
    for ln in lines:
        if ln.lower().startswith("user-agent"):
            ua = ln.split(":",1)[1].strip()
        elif ln.lower().startswith("disallow") and ua == "*":
            v = ln.split(":",1)[1].strip()
            disallows.append(v)
    path = urlparse(url).path or "/"
    for d in disallows:
        if d == "":
            continue
        if d == "/" or path.startswith(d):
            return False
    return True

def wait_for_domain(host: str, min_interval: float) -> None:
    with DOMAIN_LOCK:
        last = DOMAIN_RATE.get(host, 0)
        now = time.time()
        wait = max(0, min_interval - (now - last))
        if wait:
            time.sleep(wait)
        DOMAIN_RATE[host] = time.time()

def safe_get(url: str, delay_between_requests: float = 1.0, timeout: int = 15, ignore_robots: bool = False) -> Tuple[Optional[requests.Response], Optional[Dict[str, Any]]]:
    logger.info("GET %s", url)
    if not ignore_robots and not is_allowed(url):
        logger.warning("robots.txt запрещает %s (по возможности)", url)
        return None, {"error": "robots_disallow"}
    host = urlparse(url).netloc.lower()
    jitter = random.uniform(0.2, 0.6)
    wait_for_domain(host, delay_between_requests + jitter)
    session = get_thread_session()
    if random.random() < 0.05:
        session.headers["User-Agent"] = random.choice(USER_AGENTS)
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        snippet = (r.text or "")[:5000].lower()
        if any(x in snippet for x in ("captcha", "are you a human", "access denied", "verify you are human")):
            logger.warning("Возможна блокировка или капча на %s", url)
            return r, {"error": "blocked_or_captcha"}
        logger.info("Загружен %s (статус=%s, байт=%s)", url, r.status_code, len(r.text or ""))
        return r, None
    except requests.RequestException as e:
        logger.error("Ошибка запроса %s: %s", url, e)
        return None, {"error": str(e)}

# ----------------- Парсеры -----------------
def parse_json_ld(soup: BeautifulSoup) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            j = json.loads(script.string or "{}")
            if isinstance(j, dict):
                data.update(j)
            elif isinstance(j, list):
                for e in j:
                    if isinstance(e, dict):
                        data.update(e)
        except Exception:
            continue
    return data

def extract_attributes_from_detail_soup(soup: BeautifulSoup) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    attrs.update({f"ld_{k}": v for k, v in parse_json_ld(soup).items()})
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                k = cols[0].get_text(strip=True)
                v = cols[1].get_text(strip=True)
                if k: attrs.setdefault(k, v)
    for dl in soup.find_all("dl"):
        dt = dl.find_all("dt")
        dd = dl.find_all("dd")
        for a, b in zip(dt, dd):
            k = a.get_text(strip=True)
            v = b.get_text(strip=True)
            if k: attrs.setdefault(k, v)
    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True)
        if ":" in text:
            try:
                k, v = map(str.strip, text.split(":", 1))
                if k: attrs.setdefault(k, v)
            except Exception:
                pass
    for name in ("description", "keywords"):
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            attrs.setdefault(name, meta["content"])
    return attrs

def extract_card_info(card: BeautifulSoup, conf: Dict[str, Any]) -> Dict[str, Any]:
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

def fetch_product_detail(url: str, delay_between_requests: float = 1.0, ignore_robots: bool = False) -> Dict[str, Any]:
    if not url or url in ("Нет ссылки", None):
        return {}
    resp, err = safe_get(url, delay_between_requests=delay_between_requests, ignore_robots=ignore_robots)
    if err:
        return err
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        detail: Dict[str, Any] = {}
        h1 = soup.find("h1")
        if h1: detail["detail_title"] = h1.get_text(strip=True)
        detail.update(extract_attributes_from_detail_soup(soup))
        return detail
    except Exception as e:
        logger.exception("Ошибка парсинга для %s: %s", url, e)
        return {"error": f"parse_error: {e}"}

def process_product(base_info: Dict[str, Any], delay_details: float, ignore_robots: bool = False) -> Dict[str, Any]:
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

# ----------------- Основной скрипт -----------------
def scrape_seller(seller_url: str,
                  site_key: str,
                  max_pages: int = 5,
                  delay_pages: float = 1.0,
                  delay_details: float = 1.0,
                  max_workers: int = 4,
                  progress_callback: Optional[callable] = None,
                  ignore_robots: bool = False) -> List[Dict[str, Any]]:
    conf = SELECTORS.get(site_key)
    if not conf:
        raise ValueError("Сайт не настроен в SELECTORS")
    items: List[Dict[str, Any]] = []
    page_param_format = conf.get("page_param", "?page={page}")
    for page in range(1, max_pages+1):
        page_suffix = page_param_format.format(page=page)
        page_url = f"{seller_url.rstrip('/')}{page_suffix}"
        if progress_callback: progress_callback(f"Загрузка {page_url}")
        resp, err = safe_get(page_url, delay_between_requests=delay_pages, ignore_robots=ignore_robots)
        if err:
            if progress_callback: progress_callback(f"Ошибка страницы: {err.get('error')}")
            logger.warning("Остановка из-за ошибки страницы: %s", err)
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        tag, attrs = conf["card"]
        cards = soup.find_all(tag, attrs=attrs)
        if not cards:
            if progress_callback: progress_callback("Карточки не найдены, остановка.")
            logger.info("Карточки не найдены на странице %s", page_url)
            break
        if progress_callback: progress_callback(f"Обработка страницы {page}, карточек={len(cards)}")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_product, extract_card_info(c, conf), delay_details, ignore_robots) for c in cards]
            for f in as_completed(futures):
                try:
                    items.append(f.result())
                except Exception as e:
                    logger.exception("Ошибка товара: %s", e)
                    if progress_callback: progress_callback(f"Ошибка товара: {e}")
        time.sleep(delay_pages + random.uniform(0.1, 0.5))
    logger.info("Обработка завершена. Итемов: %d", len(items))
    return items

def detect_site_from_url(url: str) -> Optional[str]:
    dom = urlparse(url).netloc.lower()
    if "ozon." in dom: return "ozon"
    if "wildberries" in dom: return "wildberries"
    return None

# ----------------- CLI -----------------
def run_cli(argv: Optional[List[str]] = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Парсинг OZON или Wildberries")
    parser.add_argument("--url", "-u", help="URL продавца/магазина (если опущен — интерактив)")
    parser.add_argument("--site", choices=["ozon","wildberries"], help="Ключ сайта (авто-детектится если опущен)")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--delay-pages", type=float, default=1.0)
    parser.add_argument("--delay-details", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="products.xlsx")
    parser.add_argument("--ignore-robots", action="store_true", help="Игнорировать robots.txt (не рекомендуется)")
    parsed = parser.parse_args(argv)

    url = parsed.url
    if not url:
        try:
            url = input("Введите URL продавца/магазина (или оставить пустым для выхода): ").strip()
        except Exception:
            url = ""
        if not url:
            logger.info("Не введен URL — выходим.")
            print("[info] Не введен URL — выходим.")
            return

    site = parsed.site or detect_site_from_url(url)
    if not site:
        try:
            site = input("Не удалось определить сайт. Введите 'ozon' или 'wildberries': ").strip().lower()
        except Exception:
            site = ""
        if site not in ("ozon", "wildberries"):
            logger.error("Недопустимый сайт. Завершение.")
            print("[error] Недопустимый сайт. Завершение.")
            return

    def prog(m): logger.info(m); print("[+]", m)

    items = scrape_seller(url, site, max_pages=parsed.pages, delay_pages=parsed.delay_pages,
                          delay_details=parsed.delay_details, max_workers=parsed.workers,
                          progress_callback=prog, ignore_robots=parsed.ignore_robots)
    if not items:
        logger.info("Нет товаров или блокировка.")
        print("[info] Нет товаров или блокировка.")
        return
    df = pd.DataFrame(items)
    try:
        df.to_excel(parsed.out, index=False, engine="xlsxwriter")
        logger.info("Сохранено %d товаров в %s", len(df), parsed.out)
        print("[ok] Сохранено в", parsed.out)
    except Exception:
        fallback = parsed.out.rsplit(".", 1)[0] + ".json"
        with open(fallback, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.exception("Ошибка при сохранении Excel, сохранен JSON: %s", fallback)
        print("[warn] Сохранено в JSON:", fallback)

# ----------------- Точка входа -----------------
if __name__ == "__main__":
    run_cli(sys.argv[1:])
