import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib import robotparser
import time
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Константы и настройки
USER_AGENT = "Mozilla/5.0 (compatible; StreamlitBot/1.0; +https://example.com/bot)"
HEADERS = {"User-Agent": USER_AGENT}

SELECTORS = {
    "ozon": {
        "card": ("div", {"class": "b5v1"}),
        "title": ("a", {"class": "a4d3"}),
        "price": ("div", {"class": "b5v2"}),
        "img": ("img", {}),
        "base_domain": "https://ozon.ru"
    },
    "wildberries": {
        "card": ("div", {"class": "product-card"}),
        "title": ("a", {"class": "product-card__name"}),
        "price_alt": [("ins", {"class": "price__new"}), ("ins", {"class": "price__old"})],
        "img": ("img", {"class": "product-card__image"}),
        "base_domain": "https://www.wildberries.ru"
    }
}

ROBOTS_CACHE = {}
lock = threading.Lock()

# Функции для обработки robots.txt
def get_robot_parser_for(url):
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    with lock:
        if base in ROBOTS_CACHE:
            return ROBOTS_CACHE[base]
        rp = robotparser.RobotFileParser()
        try:
            rp.set_url(urljoin(base, "/robots.txt"))
            rp.read()
            ROBOTS_CACHE[base] = rp
        except Exception:
            ROBOTS_CACHE[base] = None
        return ROBOTS_CACHE[base]

def is_allowed(url):
    rp = get_robot_parser_for(url)
    if rp is None:
        return False
    return rp.can_fetch(USER_AGENT, url)

# Парсинг JSON-LD
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

# Извлечение характеристик
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

# Извлечение карточки товара
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

# Получение деталей товара
def fetch_product_detail(session, url, delay_between_requests=1.0):
    if not url or url == "Нет ссылки":
        return {}
    if not is_allowed(url):
        return {"detail_note": "robots.txt запрещает доступ"}
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        detail = {}
        if soup.find("h1"):
            detail["detail_title"] = soup.find("h1").get_text(strip=True)
        detail.update(extract_attributes_from_detail_soup(soup))
        return detail
    except Exception as e:
        return {"error": str(e)}
    finally:
        time.sleep(delay_between_requests)

# Обработка одного товара
def process_product(session, base_info, delay_details):
    if base_info.get("Ссылка") not in ("Нет ссылки", None):
        details = fetch_product_detail(session, base_info["Ссылка"], delay_between_requests=delay_details)
    else:
        details = {}
    return {**base_info, **details}

# Основная функция парсинга
def scrape_seller(seller_url, site_key, max_pages, delay_pages, delay_details, max_workers, progress_callback=None):
    conf = SELECTORS.get(site_key)
    if not conf:
        raise ValueError("Сайт не настроен в SELECTORS")
    if not is_allowed(seller_url):
        raise SystemExit(f"robots.txt не разрешает доступ к {seller_url}")

    session = requests.Session()
    session.headers.update(HEADERS)

    products = []
    for page in range(1, max_pages + 1):
        page_url = f"{seller_url.rstrip('/')}" + f"?page={page}"
        if not is_allowed(page_url):
            if progress_callback:
                progress_callback(f"robots.txt запрещает доступ к {page_url}. Остановка.")
            break
        try:
            resp = session.get(page_url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            if progress_callback:
                progress_callback(f"Ошибка загрузки страницы {page_url}: {e}")
            break

        tag, attrs = conf["card"]
        cards = soup.find_all(tag, attrs=attrs)
        if not cards:
            if progress_callback:
                progress_callback(f"На странице {page} карточки не найдены — конец.")
            break

        if progress_callback:
            progress_callback(f"Обработка страницы {page}, карточек: {len(cards)}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_product, session, extract_card_info(card, conf), delay_details)
                for card in cards
            ]
            for future in as_completed(futures):
                try:
                    product_data = future.result()
                    products.append(product_data)
                except Exception:
                    if progress_callback:
                        progress_callback("Ошибка при обработке товара")
        time.sleep(delay_pages)
    return products

# Определение сайта из URL
def detect_site_from_url(url):
    dom = urlparse(url).netloc.lower()
    if "ozon." in dom:
        return "ozon"
    if "wildberries" in dom:
        return "wildberries"
    return None

# Основная функция Streamlit
def main():
    st.title("Парсер сайтов OZON и Wildberries")
    url = st.text_input("Введите URL магазина/продавца")
    max_pages = st.number_input("Максимум страниц", min_value=1, max_value=100, value=20)
    delay_pages = st.number_input("Задержка между страницами (сек)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    delay_details = st.number_input("Задержка между запросами деталей (сек)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    max_workers = st.number_input("Потоков обработки", min_value=1, max_value=20, value=5)

    if st.button("Начать парсинг"):
        if not url:
            st.warning("Пожалуйста, введите URL.")
            return

        site_key = detect_site_from_url(url)
        if site_key:
            st.success(f"Определен сайт: {site_key}")
        else:
            site_key = st.text_input("Не удалось определить сайт. Введите 'ozon' или 'wildberries':").strip().lower()
            if site_key not in ("ozon", "wildberries"):
                st.error("Некорректный ввод сайта.")
                return

        progress_text = st.empty()
        df_container = st.empty()

        def progress_callback(msg):
            progress_text.text(msg)

        with st.spinner("Парсинг запущен..."):
            items = scrape_seller(
                url,
                site_key,
                max_pages=max_pages,
                delay_pages=delay_pages,
                delay_details=delay_details,
                max_workers=max_workers,
                progress_callback=progress_callback
            )

        if not items:
            st.info("Ничего не найдено или доступ запрещен.")
            return

        df = pd.DataFrame(items)
        st.success(f"Найдено {len(df)} товаров.")
        # Отображение таблицы
        st.dataframe(df)
        # Скачивание файла
        excel_bytes = df.to_excel(index=False, engine="xlsxwriter")
        st.download_button(
            label="Скачать Excel",
            data=pd.ExcelWriter("output.xlsx").save(),
            file_name=f"{site_key}_products.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
