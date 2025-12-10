import sys, os, subprocess, time, random, threading, json, io, logging
from urllib.parse import urljoin, urlparse

# Setup logging to console + file
LOG_FILE = "scraper.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOG_FILE, encoding="utf-8")])
logger = logging.getLogger(__name__)

# --- Dependency check + optional auto-install with captured logs ---
REQUIRED_MODULES = {"requests": "requests", "bs4": "beautifulsoup4", "pandas": "pandas",
                    "xlsxwriter": "XlsxWriter", "urllib3": "urllib3"}
missing_pkgs = []
for mod, pkg in REQUIRED_MODULES.items():
    try:
        __import__(mod)
    except Exception:
        missing_pkgs.append(pkg)

INSTALL_LOG = "install_log.txt"
if missing_pkgs:
    logger.info("Missing packages detected: %s", ", ".join(missing_pkgs))
    cmd = [sys.executable, "-m", "pip", "install"] + missing_pkgs
    try:
        logger.info("Attempting to pip install missing packages (output -> %s)...", INSTALL_LOG)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        with open(INSTALL_LOG, "wb") as f:
            f.write(proc.stdout or b"")
        logger.info("pip install finished with returncode=%s", proc.returncode)
    except subprocess.TimeoutExpired:
        with open(INSTALL_LOG, "a", encoding="utf-8") as f:
            f.write("\n[pip install timed out]\n")
        logger.error("pip install timed out; see %s", INSTALL_LOG)
    except Exception as e:
        with open(INSTALL_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[pip install error] {e}\n")
        logger.exception("pip install error: %s", e)

# Try imports (after possible install)
try:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from concurrent.futures import ThreadPoolExecutor, as_completed
except Exception as e:
    logger.exception("Missing required libraries after install attempt: %s", e)
    sys.exit(1)

# Optional Streamlit
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# ---------- Config ----------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
]
DEFAULT_HEADERS = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                   "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"}
SELECTORS = {
    "ozon": {"card": ("div", {"class": "b5v1"}), "title": ("a", {"class": "a4d3"}), "price": ("div", {"class": "b5v2"}), "img": ("img", {}), "base_domain": "https://ozon.ru"},
    "wildberries": {"card": ("div", {"class": "product-card"}), "title": ("a", {"class": "product-card__name"}),
                    "price_alt": [("ins", {"class": "price__new"}), ("ins", {"class": "price__old"})],
                    "img": ("img", {"class": "product-card__image"}), "base_domain": "https://www.wildberries.ru"},
}
RETRY_STRATEGY = Retry(total=3, status_forcelist=(429, 500, 502, 503, 504), backoff_factor=1, allowed_methods=frozenset(["GET"]))
ROBOTS_CACHE, ROBOTS_LOCK = {}, threading.Lock()
DOMAIN_RATE, DOMAIN_LOCK = {}, threading.Lock()
THREAD_LOCAL = threading.local()

# ---------- Network helpers ----------
def make_session():
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    s.headers["User-Agent"] = random.choice(USER_AGENTS)
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter); s.mount("https://", adapter)
    return s

def get_thread_session():
    if not getattr(THREAD_LOCAL, "session", None):
        THREAD_LOCAL.session = make_session()
    return THREAD_LOCAL.session

def get_robots_text(base_url):
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

def is_allowed(url):
    txt = get_robots_text(url)
    if not txt:
        return True
    lines = [l.strip() for l in txt.splitlines() if l.strip() and not l.strip().startswith("#")]
    ua = None; disallows = []
    for ln in lines:
        if ln.lower().startswith("user-agent"):
            ua = ln.split(":",1)[1].strip()
        elif ln.lower().startswith("disallow") and ua == "*":
            v = ln.split(":",1)[1].strip(); disallows.append(v)
    path = urlparse(url).path or "/"
    for d in disallows:
        if d == "": continue
        if d == "/" or path.startswith(d):
            return False
    return True

def wait_for_domain(host, min_interval):
    with DOMAIN_LOCK:
        last = DOMAIN_RATE.get(host, 0)
        now = time.time()
        wait = max(0, min_interval - (now - last))
        if wait:
            logger.debug("Waiting %.2fs for domain %s", wait, host)
            time.sleep(wait)
        DOMAIN_RATE[host] = time.time()

def safe_get(url, delay_between_requests=1.0, timeout=15):
    logger.info("GET %s", url)
    if not is_allowed(url):
        logger.warning("robots.txt disallows access to %s (best-effort)", url)
        return None, {"error": "robots.txt запрещает доступ (best-effort)"}
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
            logger.warning("Possible captcha/block detected at %s", url)
            return r, {"error": "blocked_or_captcha_detected"}
        logger.info("Loaded %s (status=%s, bytes=%s)", url, r.status_code, len(r.text or ""))
        return r, None
    except requests.RequestException as e:
        logger.error("Request failed for %s: %s", url, e)
        return None, {"error": str(e)}

# ---------- Parsing helpers ----------
def parse_json_ld(soup):
    data = {}
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

def extract_attributes_from_detail_soup(soup):
    attrs = {}
    attrs.update({f"ld_{k}": v for k, v in parse_json_ld(soup).items()})
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                k = cols[0].get_text(strip=True); v = cols[1].get_text(strip=True)
                if k: attrs.setdefault(k, v)
    for dl in soup.find_all("dl"):
        dt = dl.find_all("dt"); dd = dl.find_all("dd")
        for a,b in zip(dt,dd):
            k = a.get_text(strip=True); v = b.get_text(strip=True)
            if k: attrs.setdefault(k, v)
    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True)
        if ":" in text:
            try:
                k,v = map(str.strip, text.split(":",1))
                if k: attrs.setdefault(k, v)
            except Exception:
                pass
    for name in ("description","keywords"):
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            attrs.setdefault(name, meta["content"])
    return attrs

def extract_card_info(card, conf):
    title_tag = card.find(*conf["title"])
    title = title_tag.get_text(strip=True) if title_tag else "Нет названия"
    link = "Нет ссылки"
    if title_tag and title_tag.has_attr("href"):
        link = urljoin(conf.get("base_domain",""), title_tag["href"])
    price = "Нет цены"
    if "price" in conf:
        p = card.find(*conf["price"])
        if p: price = p.get_text(strip=True)
    elif "price_alt" in conf:
        for t,a in conf["price_alt"]:
            p = card.find(t, attrs=a)
            if p and p.get_text(strip=True):
                price = p.get_text(strip=True); break
    img = card.find(*conf["img"])
    img_url = "Нет фото"
    if img and img.has_attr("src"):
        img_url = img["src"]
        if img_url.startswith("//"): img_url = "https:" + img_url
        elif img_url.startswith("/"): img_url = urljoin(conf.get("base_domain",""), img_url)
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
        if h1: detail["detail_title"] = h1.get_text(strip=True)
        detail.update(extract_attributes_from_detail_soup(soup))
        return detail
    except Exception as e:
        logger.exception("Failed parsing detail page %s: %s", url, e)
        return {"error": f"parse_error: {e}"}

def process_product(base_info, delay_details):
    if base_info.get("Ссылка") not in ("Нет ссылки", None):
        details = fetch_product_detail(base_info["Ссылка"], delay_between_requests=delay_details)
    else:
        details = {}
    out = dict(base_info)
    if isinstance(details, dict):
        out.update(details)
    else:
        out["detail_error"] = str(details)
    return out

# ---------- Scraper ----------
def scrape_seller(seller_url, site_key, max_pages=5, delay_pages=1.0, delay_details=1.0, max_workers=4, progress_callback=None):
    conf = SELECTORS.get(site_key)
    if not conf:
        raise ValueError("Site not configured in SELECTORS")
    items = []
    for page in range(1, max_pages+1):
        page_url = f"{seller_url.rstrip('/')}" + f"?page={page}"
        if progress_callback: progress_callback(f"Loading {page_url}")
        resp, err = safe_get(page_url, delay_between_requests=delay_pages)
        if err:
            if progress_callback: progress_callback(f"Page error: {err.get('error')}")
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        tag, attrs = conf["card"]
        cards = soup.find_all(tag, attrs=attrs)
        if not cards:
            if progress_callback: progress_callback("No cards found, stop.")
            break
        if progress_callback: progress_callback(f"Processing page {page}, cards={len(cards)}")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_product, extract_card_info(c, conf), delay_details) for c in cards]
            for f in as_completed(futures):
                try:
                    items.append(f.result())
                except Exception as e:
                    logger.exception("Product processing error: %s", e)
                    if progress_callback: progress_callback(f"Product error: {e}")
        time.sleep(delay_pages + random.uniform(0.1,0.5))
    logger.info("Scraping complete. Total items: %d", len(items))
    return items

def detect_site_from_url(url):
    dom = urlparse(url).netloc.lower()
    if "ozon." in dom: return "ozon"
    if "wildberries" in dom: return "wildberries"
    return None

# ---------- CLI and Streamlit UI ----------
def run_cli(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="Scrape OZON or Wildberries seller pages")
    parser.add_argument("url", nargs="?", help="Seller/shop URL (omitted -> interactive prompt)")
    parser.add_argument("--site", choices=["ozon","wildberries"], help="Site key (auto-detected if omitted)")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--delay-pages", type=float, default=1.0)
    parser.add_argument("--delay-details", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="products.xlsx")
    parsed = parser.parse_args(args=args)

    url = parsed.url
    if not url:
        try:
            url = input("Enter seller/shop URL (leave blank to exit): ").strip()
        except Exception:
            url = ""
        if not url:
            logger.info("No URL entered; exiting without error.")
            return

    site = parsed.site or detect_site_from_url(url)
    if not site:
        try:
            site = input("Could not detect site. Enter 'ozon' or 'wildberries' (leave blank to exit): ").strip().lower()
        except Exception:
            site = ""
        if site not in ("ozon","wildberries"):
            logger.error("Invalid or missing site; exiting.")
            return

    def prog(m):
        logger.info(m); print(m)
    items = scrape_seller(url, site, max_pages=parsed.pages, delay_pages=parsed.delay_pages,
                          delay_details=parsed.delay_details, max_workers=parsed.workers, progress_callback=prog)
    if not items:
        logger.info("No items scraped or blocked.")
        return
    df = pd.DataFrame(items)
    logger.info("Saving %d items to %s", len(df), parsed.out)
    try:
        df.to_excel(parsed.out, index=False, engine="xlsxwriter")
        logger.info("Saved Excel: %s", parsed.out)
    except Exception as e:
        fallback = parsed.out.rsplit(".",1)[0] + ".json"
        with open(fallback, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.exception("Failed to save Excel; saved JSON to %s", fallback)

def run_streamlit():
    st.title("Улучшенный парсер OZON и Wildberries")
    url = st.text_input("Seller/shop URL")
    site_choice = st.selectbox("Site (auto)", options=["auto","ozon","wildberries"])
    max_pages = st.number_input("Max pages", min_value=1, max_value=200, value=3)
    delay_pages = st.number_input("Delay between pages (s)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    delay_details = st.number_input("Delay between details (s)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    workers = st.number_input("Threads", min_value=1, max_value=20, value=4)
    if st.button("Start"):
        if not url:
            st.warning("Enter URL"); return
        site = None if site_choice == "auto" else site_choice
        if site is None:
            site = detect_site_from_url(url)
            if not site:
                st.error("Could not detect site. Choose manually."); return
        progress_area = st.empty()
        def progress_callback(m):
            logger.info(m)
            progress_area.text(m)
        with st.spinner("Scraping..."):
            items = scrape_seller(url, site, max_pages=int(max_pages), delay_pages=float(delay_pages),
                                  delay_details=float(delay_details), max_workers=int(workers),
                                  progress_callback=progress_callback)
        if not items:
            st.info("Nothing found or blocked."); return
        df = pd.DataFrame(items)
        st.success(f"Found {len(df)} items")
        st.dataframe(df)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="products")
        buf.seek(0)
        st.download_button("Download Excel", data=buf.getvalue(), file_name="products.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--gui" in argv:
        if not STREAMLIT_AVAILABLE:
            logger.error("Streamlit is not installed. Install with: python -m pip install streamlit")
            sys.exit(1)
        logger.info("To run the GUI: streamlit run %s", os.path.abspath(__file__))
        print("To run the GUI: streamlit run", os.path.abspath(__file__))
    elif STREAMLIT_AVAILABLE and ("streamlit" in " ".join(sys.argv) or os.environ.get("STREAMLIT_RUNNING") or os.getenv("STREAMLIT_SERVER_RUNNING")):
        run_streamlit()
    else:
        run_cli(argv)
