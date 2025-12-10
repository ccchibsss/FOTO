# streamlit_scraper_full_commented.py
"""
Streamlit app: robust OZON / Wildberries seller scraper with comments.

Usage:
  streamlit run streamlit_scraper_full_commented.py

Features (concise):
- Simple dependency check (shows helpful message inside Streamlit).
- Thread-local requests.Session with retry adapter.
- Per-domain rate limiting + jitter to be polite.
- Basic robots.txt "best-effort" check (cached).
- Extracts card-level info (title, link, price, image) and optionally loads product details.
- Parallel fetching of product detail pages with ThreadPoolExecutor.
- Download results as XLSX or JSON.
- Defensive error handling and progress log.

Notes:
- The CSS selectors are examples and may need updates per real site HTML.
- Respect site ToS and robots.txt.
"""

from __future__ import annotations
import time
import random
import threading
import json
import io
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urljoin, urlparse

# ---- Imports and dependency check ----
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
    # If streamlit isn't available this script won't run as an app;
    # print short instruction and stop execution inside streamlit (if started).
    msg = f"Missing dependency (run outside Streamlit or install): {e}"
    try:
        # If running under streamlit import failed, show in terminal and stop.
        print(msg)
    except Exception:
        pass
    raise

# ---- Constants / config (edit selectors as needed) ----
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
]
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}
# Example selectors; real sites change frequently -> adjust before use.
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

# Retry configuration for requests Session
RETRY_STRATEGY = Retry(total=3, status_forcelist=(429, 500, 502, 503, 504),
                       backoff_factor=1, allowed_methods=frozenset({"GET"}))

# Thread-local session, robots cache, domain rate limiting
THREAD_LOCAL = threading.local()
ROBOTS_CACHE: Dict[str, Optional[RobotFileParser]] = {}
ROBOTS_LOCK = threading.Lock()
DOMAIN_RATE: Dict[str, float] = {}
DOMAIN_LOCK = threading.Lock()

# ---- Network helpers ----
def make_session() -> requests.Session:
    """Create a requests.Session with retry adapter and default headers."""
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    s.headers["User-Agent"] = random.choice(USER_AGENTS)
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter); s.mount("https://", adapter)
    return s

def get_thread_session() -> requests.Session:
    """Return a thread-local session (one per worker thread)."""
    if not getattr(THREAD_LOCAL, "session", None):
        THREAD_LOCAL.session = make_session()
    return THREAD_LOCAL.session

def get_robot_parser(base_url: str) -> Optional[RobotFileParser]:
    """
    Return cached RobotFileParser for a base URL. If robots.txt can't be fetched,
    cache None and allow fetching (best-effort).
    """
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
    """Check robots.txt (best-effort). If robots cannot be read, return True."""
    rp = get_robot_parser(url)
    if rp is None:
        return True
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True

def wait_for_domain(host: str, min_interval: float) -> None:
    """Ensure at least min_interval seconds between requests to same host."""
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
    Perform GET with:
      - optional robots.txt respect (best-effort)
      - per-domain rate limiting + jitter
      - UA rotation
      - simple detection of blocks/captchas
    Returns (response, None) on success, or (response_or_none, {"error": ...}) on problem.
    """
    if not ignore_robots and not allowed_by_robots(url):
        return None, {"error": "robots_disallow"}
    host = urlparse(url).netloc.lower()
    wait_for_domain(host, delay_between_requests + random.uniform(0.2, 0.6))
    session = get_thread_session()
    # small chance to rotate UA for this session
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

# ---- Parsing helpers ----
def parse_json_ld(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract data from <script type='application/ld+json'> elements."""
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
    """Try many common HTML patterns to extract key/value attributes from a detail page."""
    attrs: Dict[str, Any] = {}
    attrs.update({f"ld_{k}": v for k, v in parse_json_ld(soup).items()})
    # Table rows
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                k = cols[0].get_text(strip=True); v = cols[1].get_text(strip=True)
                if k: attrs.setdefault(k, v)
    # Definition lists
    for dl in soup.find_all("dl"):
        dt = dl.find_all("dt"); dd = dl.find_all("dd")
        for a, b in zip(dt, dd):
            k = a.get_text(strip=True); v = b.get_text(strip=True)
            if k: attrs.setdefault(k, v)
    # List items with "key: value"
    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True)
        if ":" in text:
            try:
                k, v = map(str.strip, text.split(":", 1))
                if k: attrs.setdefault(k, v)
            except Exception:
                pass
    # Meta description/keywords
    for name in ("description", "keywords"):
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            attrs.setdefault(name, meta["content"])
    return attrs

def extract_card_info(card: BeautifulSoup, conf: Dict[str, Any]) -> Dict[str, str]:
    """Extracts title, link, price, image from a product card element using conf selectors."""
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
    """Fetch a product detail page and extract attributes. Returns dict or error dict."""
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
    """Wrapper for parallel fetching of product details and merging with base info."""
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

# ---- Scraper core ----
def detect_site_from_url(url: str) -> Optional[str]:
    dom = urlparse(url).netloc.lower()
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
    """
    Iterate pages, parse cards, and fetch details in parallel.
    Returns list of item dicts.
    """
    conf = SELECTORS.get(site_key)
    if not conf:
        raise ValueError("Site not configured in SELECTORS")
    items: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        page_suffix = conf.get("page_param", "?page={page}").format(page=page)
        page_url = f"{seller_url.rstrip('/')}{page_suffix}"
        if progress_callback: progress_callback(f"Loading {page_url}")
        resp, err = safe_get(page_url, delay_between_requests=delay_pages, ignore_robots=ignore_robots)
        if err:
            if progress_callback: progress_callback(f"Page error: {err.get('error')}")
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        tag, attrs = conf["card"]
        cards = soup.find_all(tag, attrs=attrs)
        if not cards:
            if progress_callback: progress_callback("No cards found, stopping pagination.")
            break
        if progress_callback: progress_callback(f"Found {len(cards)} cards on page {page}")
        # Parallel detail fetching
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_product, extract_card_info(c, conf), delay_details, ignore_robots) for c in cards]
            for f in as_completed(futures):
                try:
                    res = f.result()
                    items.append(res)
                except Exception as e:
                    if progress_callback: progress_callback(f"Product error: {e}")
        # polite pause
        time.sleep(delay_pages + random.uniform(0.05, 0.4))
    return items

# ---- Streamlit UI ----
st.set_page_config(page_title="OZON/Wildberries Scraper", layout="wide")
st.title("OZON / Wildberries Scraper (Streamlit)")

# Input controls
col1, col2 = st.columns([2, 1])
with col1:
    seller_url = st.text_input("Seller/shop URL")
    site_choice = st.selectbox("Site", options=["auto", "ozon", "wildberries"], index=0)
    max_pages = st.number_input("Max pages", min_value=1, max_value=200, value=3)
    delay_pages = st.number_input("Delay between pages (s)", min_value=0.0, value=1.0, step=0.1)
    delay_details = st.number_input("Delay between details (s)", min_value=0.0, value=1.0, step=0.1)
    workers = st.number_input("Threads", min_value=1, max_value=20, value=4)
    ignore_robots = st.checkbox("Ignore robots.txt (not recommended)", value=False)
with col2:
    out_name = st.text_input("Output filename", value="products.xlsx")
    start_btn = st.button("Start scraping")
    st.markdown("Notes:")
    st.markdown("- Selectors are examples; adapt for current site layout.")
    st.markdown("- Use small page counts & delays to avoid being blocked.")

log_box = st.empty()
result_box = st.empty()

def append_log(msg: str) -> None:
    """Append message to a session_state log and display in log_box."""
    if "log" not in st.session_state:
        st.session_state["log"] = []
    st.session_state["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    log_box.text_area("Progress log", value="\n".join(st.session_state["log"]), height=240)

if start_btn:
    # reset logs
    st.session_state["log"] = []
    if not seller_url:
        st.error("Please enter seller/shop URL")
    else:
        # detect site if needed
        site = None if site_choice == "auto" else site_choice
        if site is None:
            site = detect_site_from_url(seller_url)
        if not site:
            st.error("Could not detect site from URL. Choose 'ozon' or 'wildberries'.")
        else:
            append_log(f"Starting scrape for site='{site}' url='{seller_url}'")
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
                append_log("No items found or scraping failed/blocked.")
                st.info("No items found or scraping failed/blocked. See log.")
            else:
                df = pd.DataFrame(items)
                append_log(f"Scraped {len(df)} items; preparing output.")
                result_box.dataframe(df)
                # prepare excel in-memory
                buf = io.BytesIO()
                try:
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        df.to_excel(writer, index=False, sheet_name="products")
                    buf.seek(0)
                    st.download_button("Download Excel", data=buf.getvalue(), file_name=out_name,
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception:
                    # fallback to JSON download
                    st.download_button("Download JSON", data=json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8"),
                                       file_name=out_name.rsplit(".", 1)[0] + ".json", mime="application/json")
                append_log("Done. You may download results.")
