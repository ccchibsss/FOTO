import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib import robotparser
import time
import json
import pandas as pd

USER_AGENT = "Mozilla/5.0 (compatible; SafeBot/1.0; +https://example.com/bot)"
HEADERS = {"User-Agent": USER_AGENT}

# Селекторы для распознавания карточек на странице продавца (можно расширять)
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

# Кэш robots для доменов
ROBOTS_CACHE = {}

def get_robot_parser_for(url):
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
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
        # Если robots.txt недоступен — по безопасности останавливаем сбор
        return False
    return rp.can_fetch(USER_AGENT, url)

def parse_json_ld(soup):
    """Попытка извлечь структурированные данные из JSON-LD"""
    data = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            j = json.loads(script.string or "{}")
            if isinstance(j, dict):
                # часто в ld присутствует '@type': 'Product'
                if j.get("@type", "").lower() == "product" or "name" in j:
                    data.update(j)
            elif isinstance(j, list):
                for elem in j:
                    if isinstance(elem, dict) and (elem.get("@type","").lower()=="product" or "name" in elem):
                        data.update(elem)
        except Exception:
            continue
    return data

def extract_attributes_from_detail_soup(soup):
    """Извлекает таблицы/списки характеристик и метаданные со страницы товара"""
    attrs = {}

    # JSON-LD
    attrs.update({f"ld_{k}": v for k, v in parse_json_ld(soup).items()})

    # Таблицы ключ-значение
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                k = cols[0].get_text(strip=True)
                v = cols[1].get_text(strip=True)
                if k:
                    attrs.setdefault(k, v)

    #списки dl dt/dd
    for dl in soup.find_all("dl"):
        dt = dl.find_all("dt")
        dd = dl.find_all("dd")
        for a, b in zip(dt, dd):
            k = a.get_text(strip=True)
            v = b.get_text(strip=True)
            if k:
                attrs.setdefault(k, v)

    # списки li где есть ":"
    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True)
        if ":" in text:
            try:
                k, v = map(str.strip, text.split(":", 1))
                if k:
                    attrs.setdefault(k, v)
            except Exception:
                continue

    # Метатеги
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

    # Цена
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

    # Фото
    img = card.find(*conf["img"])
    img_url = img.get("src") if img and img.has_attr("src") else "Нет фото"
    if img_url.startswith("//"):
        img_url = "https:" + img_url
    elif img_url.startswith("/"):
        img_url = urljoin(conf.get("base_domain", ""), img_url)

    return {"Название": title, "Ссылка": link, "Цена": price, "Фото": img_url}

def fetch_product_detail(session, url, delay_between_requests=1.0):
    detail = {}
    if not url or url == "Нет ссылки":
        return detail
    if not is_allowed(url):
        # по безопасности не посещаем запрещённые URL
        return {"detail_note": "robots.txt запрещает доступ"}
    try:
        resp = session.get(url, timeout=10, headers=HEADERS)
        resp.raise_for_status()
    except Exception as e:
        return {"detail_error": str(e)}
    soup = BeautifulSoup(resp.text, "html.parser")
    #Название со страницы (если отличается), краткое описание, характеристики
    page_title = soup.find("h1")
    if page_title:
        detail["detail_title"] = page_title.get_text(strip=True)
    detail_attrs = extract_attributes_from_detail_soup(soup)
    detail.update(detail_attrs)
    time.sleep(delay_between_requests)
    return detail

def scrape_seller(seller_url, site_key, max_pages=50, delay_pages=1.0, delay_details=1.0, follow_details=True):
    conf = SELECTORS.get(site_key)
    if not conf:
        raise ValueError("Сайт не настроен в SELECTORS")
    if not is_allowed(seller_url):
        raise SystemExit(f"robots.txt не разрешает доступ к {seller_url} — прекращаю работу.")

    session = requests.Session()
    session.headers.update(HEADERS)

    products = []
    for page in range(1, max_pages + 1):
        page_url = f"{seller_url.rstrip('/')}" + f"?page={page}"
        if not is_allowed(page_url):
            print(f"[{page}] robots.txt запрещает доступ к {page_url}. Останавливаемся.")
            break
        try:
            resp = session.get(page_url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"[{page}] Ошибка запроса {page_url}: {e}")
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        tag, attrs = conf["card"]
        cards = soup.find_all(tag, attrs=attrs)
        if not cards:
            print(f"[{page}] Товары не найдены - возможно конец списка. Останавливаемся.")
            break
        print(f"[{page}] Найдено карточек: {len(cards)}. Обрабатываю...")
        for card in cards:
            try:
                base_info = extract_card_info(card, conf)
                if follow_details and base_info.get("Ссылка") not in ("Нет ссылки", None):
                    details = fetch_product_detail(session, base_info["Ссылка"], delay_between_requests=delay_details)
                else:
                    details = {}
                row = {**base_info, **details}
                products.append(row)
            except Exception:
                continue
        time.sleep(delay_pages)
    return products

def detect_site_from_url(url):
    dom = urlparse(url).netloc.lower()
    if "ozon." in dom:
        return "ozon"
    if "wildberries" in dom:
        return "wildberries"
    return None

if __name__ == "__main__":
    seller_url = input("Введите URL продавца (например, страница магазина на OZON/Wildberries): ").strip()
    if not seller_url:
        raise SystemExit("URL не введён. Выход.")

    detected = detect_site_from_url(seller_url)
    if detected:
        print(f"Авто-определён сайт: {detected}")
        site_key = detected
    else:
        choice = input("Не могу определить маркетплейс. Введите 'ozon' или 'wildberries': ").strip().lower()
        if choice not in ("ozon", "wildberries"):
            raise SystemExit("Неверный выбор сайта. Выход.")
        site_key = choice

    max_pages = int(input("Максимум страниц для обхода (по умолчанию 20): ") or "20")
    follow = input("Переходить на страницы товаров для дополнительных характеристик? (y/N): ").strip().lower() == "y"
    delay_pages = float(input("Задержка между страницами в секундах (по умолчанию 1.0): ") or "1.0")
    delay_details = float(input("Задержка между запросами деталей в секундах (по умолчанию 1.0): ") or "1.0")

    print("Запуск парсинга (строго по robots.txt)...")
    items = scrape_seller(seller_url, site_key, max_pages=max_pages,
                          delay_pages=delay_pages, delay_details=delay_details,
                          follow_details=follow)

    df = pd.DataFrame(items)
    if df.empty:
        print("Ничего не найдено или доступ запрещён robots.txt.")
    else:
        out_file = f"{site_key}_seller_products.xlsx"
        df.to_excel(out_file, index=False, engine="xlsxwriter")
        print(f"Сохранено {len(df)} записей в {out_file}")
