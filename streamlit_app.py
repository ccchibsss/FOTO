# streamlit_scraper_final.py
"""
Resilient Streamlit scraper for OZON / Wildberries (fixed).

- Run with: streamlit run streamlit_scraper_final.py
- If run with plain `python`, the script will print friendly instructions and exit (no traceback).
- Does NOT attempt automatic package installation.
- If pandas is missing the app still works and offers JSON/CSV downloads.
- Polite per-domain rate limiting, UA rotation, basic robots.txt best-effort (not strict).
- Adjust SELECTORS to match real site HTML before scraping.
"""

from __future__ import annotations
import sys
import time
import random
import threading
import json
import io
import csv
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse

# If Streamlit is not available (running with plain python), print instructions
# and exit cleanly.
try:
    import streamlit as st
except Exception:
    print("This script is intended to be run with Streamlit.")
    print("Install Streamlit and required packages and run:")
    print("  python -m pip install streamlit requests beautifulsoup4")
    print("  streamlit run streamlit_scraper_final.py")
    # Exit without stack trace
    sys.exit(0)

# Optional dependencies: requests and bs4 are required for scraping; pandas
# optional.
_missing = []
try:
    import requests
except Exception:
    _missing.append("requests")
try:
    from bs4 import BeautifulSoup
except Exception:
    _missing.append("beautifulsoup4 (bs4)")
try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

if _missing:
    st.error("Missing required packages: " + ", ".join(_missing))
    st.info("Install them in your environment, then restart Streamlit, e.g.:")
    st.code("python -m pip install " + " ".join(_missing))
    st.stop()

# --- Config (edit selectors for actual site HTML) ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/116.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/115.0 Safari/537.36",
]
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}
SELECTORS = {
    "ozon": {
        "card": ("div", {"class": "b5v1"}),      # example — update for real site
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

# Thread-local session and simple caches
THREAD_LOCAL = threading.local()
ROBOTS_CACHE: Dict[str, str] = {}
ROBOTS_LOCK = threading.Lock()
DOMAIN_RATE: Dict[str, float] = {}
DOMAIN_LOCK = threading.Lock()


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    s.headers["User-Agent"] = random.choice(USER_AGENTS)
    return s


def get_thread_session() -> requests.Session:
    if not getattr(THREAD_LOCAL, "session", None):
        THREAD_LOCAL.session = make_session()
    return THREAD_LOCAL.session


def fetch_robots_text(base: str) -> str:
    with ROBOTS_LOCK:
        cached = ROBOTS_CACHE.get(base)
    if cached is not None:
        return cached
    try:
        r = requests.get(urljoin(base, "/robots.txt"), headers={"User-Agent": USER_AGENTS[0]}, timeout=5)
        text = r.text if r.status_code == 200 else ""
    except Exception:
        text = ""
    with ROBOTS_LOCK:
        ROBOTS_CACHE[base] = text
    return text


def polite_wait_for_domain(host: str, min_interval: float) -> None:
    with DOMAIN_LOCK:
        last = DOMAIN_RATE.get(host, 0.0)
        now = time.time()
        wait = max(0.0, min_interval - (now - last))
        if wait:
            time.sleep(wait)
        DOMAIN_RATE[host] = time.time()


def safe_get(url: str, delay_between_requests: float = 1.0, timeout: int = 15, ignore_robots: bool = False) -> Tuple[Optional[requests.Response], Optional[Dict[str, str]]]:
    try:
        parsed = urlparse(url)
    except Exception:
        return None, {"error": "invalid_url"}
    base = f"{parsed.scheme}://{parsed.netloc}"
    if not ignore_robots:
        robots_txt = fetch_robots_text(base)
        if robots_txt:
            lines = [l.strip() for l in robots_txt.splitlines() if l.strip()]
            ua = None
            disallows: List[str] = []
            for ln in lines:
                if ln.lower().startswith("user-agent"):
                    ua = ln.split(":", 1)[1].strip()
                if ln.lower().startswith("disallow") and ua == "*":
                    v = ln.split(":", 1)[1].strip()
                    disallows.append(v)
            path = parsed.path or "/"
            for d in disallows:
                if d == "/" or (d and path.startswith(d)):
                    return None, {"error": "robots_disallow"}
    host = parsed.netloc.lower()
    polite_wait_for_domain(host, delay_between_requests + random.uniform(0.2, 0.6))
    session = get_thread_session()
    if random.random() < 0.05:
        session.headers["User-Agent"] = random.choice(USER_AGENTS)
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        snippet = (r.text or "")[:5000].lower()
        if any(token in snippet for token in ("captcha", "are you a human", "access denied", "verify you are human")):
            return r, {"error": "blocked_or_captcha"}
        return r, None
    except requests.RequestException as e:
        return None, {"error": str(e)}


def parse_json_ld(soup: BeautifulSoup) -> Dict[str, Any]:
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
    attrs: Dict[str, Any] = {}
    attrs.update({f"ld_{k}": v for k, v in parse_json_ld(soup).items()})
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                k = cols[0].get_text(strip=True); v = cols[1].get_text(strip=True)
                if k: attrs.setdefault(k, v)
    for dl in soup.find_all("dl"):
        dt = dl.find_all("dt"); dd = dl.find_all("dd")
        for a, b in zip(dt, dd):
            k = a.get_text(strip=True); v = b.get_text(strip=True)
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


def extract_card_info(card: BeautifulSoup, conf: Dict[str, Any]) -> Dict[str, str]:
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
                price = p.get_text(strip=True); break
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
        return {"error": f"parse_error: {e}"}


from concurrent.futures import ThreadPoolExecutor, as_completed


def process_product(base_info: Dict[str, str], delay_details: float, ignore_robots: bool = False) -> Dict[str, Any]:
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


def detect_site_from_url(url: str) -> Optional[str]:
    try:
        dom = urlparse(url).netloc.lower()
    except Exception:
        return None
    if "ozon." in dom: return "ozon"
    if "wildberries" in dom or "wbx" in dom: return "wildberries"
    return None


def scrape_seller(seller_url: str,
                  site_key: str,
                  max_pages: int = 3,
                  delay_pages: float = 1.0,
                  delay_details: float = 1.0,
                  max_workers: int = 4,
                  progress_callback: Optional[callable] = None,
                  ignore_robots: bool = False) -> List[Dict[str, Any]]:
    conf = SELECTORS.get(site_key)
    if not conf:
        raise ValueError("Site not configured")
    items: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        page_param = conf.get("page_param", "?page={page}").format(page=page)
        page_url = f"{seller_url.rstrip('/')}{page_param}"
        if progress_callback: progress_callback(f"Loading {page_url}")
        resp, err = safe_get(page_url, delay_between_requests=delay_pages, ignore_robots=ignore_robots)
        if err:
            if progress_callback: progress_callback(f"Page error: {err.get('error')}")
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        tag, attrs = conf["card"]
        cards = soup.find_all(tag, attrs=attrs)
        if not cards:
            if progress_callback: progress_callback("No cards found on page; stopping.")
            break
        if progress_callback: progress_callback(f"Found {len(cards)} cards on page {page}")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_product, extract_card_info(c, conf), delay_details, ignore_robots) for c in cards]
            for f in as_completed(futures):
                try:
                    items.append(f.result())
                except Exception as e:
                    if progress_callback: progress_callback(f"Product error: {e}")
        time.sleep(delay_pages + random.uniform(0.05, 0.4))
    return items


# --- Streamlit UI ---
st.set_page_config(page_title="OZON/WB Scraper", layout="wide")
st.title("OZON / Wildberries Scraper (resilient)")

col1, col2 = st.columns([2, 1])
with col1:
    seller_url = st.text_input("Seller/shop URL")
    site_choice = st.selectbox("Site (auto-detect)", ["auto", "ozon", "wildberries"])
    max_pages = st.number_input("Max pages", min_value=1, max_value=200, value=3)
    delay_pages = st.number_input("Delay between pages (s)", min_value=0.0, value=1.0, step=0.1)
    delay_details = st.number_input("Delay between details (s)", min_value=0.0, value=1.0, step=0.1)
    workers = st.number_input("Threads", min_value=1, max_value=20, value=4)
    ignore_robots = st.checkbox("Ignore robots.txt (not recommended)", value=False)
with col2:
    out_name = st.text_input("Output filename", value="products.xlsx")
    start = st.button("Start scraping")
    st.markdown("Notes:")
    st.markdown("- SELECTORS are sample values; update them for the current site layout.")
    st.markdown("- Use small page counts & delays to avoid being blocked; respect ToS and robots.txt.")
    if not PANDAS_AVAILABLE:
        st.warning("pandas not available — Excel download disabled. JSON/CSV downloads provided.")

log_box = st.empty()
result_box = st.empty()


def append_log(msg: str) -> None:
    if "log" not in st.session_state:
        st.session_state["log"] = []
    st.session_state["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    log_box.text_area("Progress log", value="\n".join(st.session_state["log"]), height=240)


if start:
    st.session_state["log"] = []
    if not seller_url:
        st.error("Please enter seller/shop URL")
    else:
        site = None if site_choice == "auto" else site_choice
        if site is None:
            site = detect_site_from_url(seller_url)
        if not site:
            st.error("Could not detect site from URL; choose manually")
        else:
            append_log(f"Starting scrape: site={site}, url={seller_url}")
            with st.spinner("Scraping..."):
                try:
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
                    append_log(f"Fatal error: {e}")
                    st.exception(e)
                    items = []
            if not items:
                append_log("No items found or scraping blocked.")
                st.info("No items found or blocked. See log.")
            else:
                append_log(f"Scraped {len(items)} items.")
                result_box.dataframe(items)
                # JSON download
                json_bytes = json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button("Download JSON", data=json_bytes,
                                   file_name=(out_name.rsplit(".", 1)[0] + ".json"),
                                   mime="application/json")
                # CSV download
                keys: List[str] = []
                for it in items:
                    for k in it.keys():
                        if k not in keys:
                            keys.append(k)
                csv_buf = io.StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                for row in items:
                    writer.writerow({k: (row.get(k, "") if row.get(k, "") is not None else "") for k in keys})
                st.download_button("Download CSV", data=csv_buf.getvalue().encode("utf-8"),
                                   file_name=(out_name.rsplit(".", 1)[0] + ".csv"),
                                   mime="text/csv")
                # Excel (if pandas available)
                if PANDAS_AVAILABLE:
                    try:
                        excel_buf = io.BytesIO()
                        df = pd.DataFrame(items)
                        with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                            df.to_excel(writer, index=False, sheet_name="products")
                        excel_buf.seek(0)
                        st.download_button("Download Excel (xlsx)", data=excel_buf.getvalue(),
                                           file_name=out_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    except Exception as e:
                        append_log(f"Excel generation failed: {e}")
                        st.warning("Excel generation failed; use JSON/CSV fallback.")
                append_log("Finished. Downloads ready.")
