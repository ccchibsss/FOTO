import sys
import os
import time
import random
import threading
import json
import io
from urllib.parse import urljoin, urlparse

# Optional GUI
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# Required libs; if missing, print instructions and exit gracefully
try:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from concurrent.futures import ThreadPoolExecutor, as_completed
except Exception:
    print("Missing required packages. Install with:\n  python -m pip install requests beautifulsoup4 pandas xlsxwriter urllib3")
    sys.exit(1)

# ---------- Config ----------
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
    },
    "wildberries": {
        "card": ("div", {"class": "product-card"}),
        "title": ("a", {"class": "product-card__name"}),
        "price_alt": [("ins", {"class": "price__new"}), ("ins", {"class": "price__old"})],
        "img": ("img", {"class": "product-card__image"}),
        "base_domain": "https://www.wildberries.ru",
    },
}
RETRY_STRATEGY = Retry(total=3, status_forcelist=(429, 500, 502, 503, 504), backoff_factor=1, allowed_methods=frozenset(["GET"]))
ROBOTS_CACHE = {}
ROBOTS_LOCK = threading.Lock()
DOMAIN_RATE = {}
DOMAIN_LOCK = threading.Lock()
THREAD_LOCAL = threading.local()

# ---------- Network helpers ----------
def make_session():
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    s.headers["User-Agent"] = random.choice(USER_AGENTS)
    adapter = HTTPAdapter(max_retries=RETRY_STRATEGY, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
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

def wait_for_domain(host, min_interval):
    with DOMAIN_LOCK:
        last = DOMAIN_RATE.get(host, 0)
        now = time.time()
        wait = max(0, min_interval - (now - last))
        if wait:
            time.sleep(wait)
        DOMAIN_RATE[host] = time.time()

def safe_get(url, delay_between_requests=1.0, timeout=15):
    if not is_allowed(url):
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
        snippet = r.text[:5000].lower()
        if any(x in snippet for x in ("captcha", "are you a human", "access denied", "verify you are human")):
            return r, {"error": "blocked_or_captcha_detected"}
        return r, None
    except requests.RequestException as e:
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
                    if progress_callback: progress_callback(f"Product error: {e}")
        time.sleep(delay_pages + random.uniform(0.1,0.5))
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
            url = input("Enter seller/shop URL: ").strip()
        except Exception:
            print("No URL provided. Exiting.")
            return
        if not url:
            print("No URL entered. Exiting.")
            return

    site = parsed.site or detect_site_from_url(url)
    if not site:
        try:
            site = input("Could not detect site. Enter 'ozon' or 'wildberries': ").strip().lower()
        except Exception:
            print("Site not provided. Exiting.")
            return
        if site not in ("ozon","wildberries"):
            print("Invalid site. Exiting.")
            return

    def prog(m): print("[+]", m)
    items = scrape_seller(url, site, max_pages=parsed.pages, delay_pages=parsed.delay_pages,
                          delay_details=parsed.delay_details, max_workers=parsed.workers, progress_callback=prog)
    if not items:
        print("No items scraped or blocked.")
        return
    df = pd.DataFrame(items)
    print(f"Scraped {len(df)} items. Saving to {parsed.out} ...")
    try:
        df.to_excel(parsed.out, index=False, engine="xlsxwriter")
        print("Saved:", parsed.out)
    except Exception as e:
        fallback = parsed.out.rsplit(".",1)[0] + ".json"
        with open(fallback, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print("Failed to save Excel, saved JSON to", fallback, "error:", e)

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
            st.warning("Enter URL")
            return
        site = None if site_choice == "auto" else site_choice
        if site is None:
            site = detect_site_from_url(url)
            if not site:
                st.error("Could not detect site. Choose manually.")
                return
        prog_area = st.empty()
        def progress_callback(m): prog_area.text(m)
        with st.spinner("Scraping..."):
            items = scrape_seller(url, site, max_pages=int(max_pages), delay_pages=float(delay_pages),
                                  delay_details=float(delay_details), max_workers=int(workers),
                                  progress_callback=progress_callback)
        if not items:
            st.info("Nothing found or blocked.")
            return
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
    args = sys.argv[1:]
    if "--gui" in args:
        if not STREAMLIT_AVAILABLE:
            print("Streamlit is not installed. Install with: python -m pip install streamlit")
            sys.exit(1)
        # When running with `python script.py --gui` we can't start Streamlit server here;
        # just inform user how to run it.
        print("To run the GUI: streamlit run " + os.path.abspath(__file__))
    elif STREAMLIT_AVAILABLE and ("streamlit" in " ".join(sys.argv) or os.environ.get("STREAMLIT_RUNNING") or os.getenv("STREAMLIT_SERVER_RUNNING")):
        # Likely invoked via `streamlit run`, start the app
        run_streamlit()
    else:
        # CLI mode (interactive fallback if URL omitted)
        run_cli(args=args)
