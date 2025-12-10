import streamlit as st
import pandas as pd
import sqlite3
import time
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from playwright.sync_api import sync_playwright

# Логирование
logging.basicConfig(level=logging.INFO)

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('ozon_extended.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            title TEXT,
            price TEXT,
            link TEXT,
            price_num REAL,
            image_url TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS characteristics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            key TEXT,
            value TEXT,
            FOREIGN KEY(product_id) REFERENCES products(id)
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_query ON products(query)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_title ON products(title)')
    conn.commit()
    conn.close()

init_db()

# --- Кеширование извлечения характеристик и изображений ---
@lru_cache(maxsize=1000)
def fetch_product_details_cached(product_url):
    return fetch_product_details(product_url)

def fetch_product_details(product_url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(product_url, timeout=60000)
            time.sleep(2)
            # Извлечение изображения
            try:
                image_url = page.query_selector("img[data-testid='product-image']").get_attribute('src')
            except:
                image_url = None
            # Извлечение характеристик
            characteristics = {}
            try:
                rows = page.query_selector_all("div[data-testid='product-characteristics'] > div")
                for row in rows:
                    key = row.query_selector("div:nth-child(1)").inner_text()
                    value = row.query_selector("div:nth-child(2)").inner_text()
                    characteristics[key] = value
            except:
                pass
            browser.close()
            return image_url, characteristics
    except Exception as e:
        logging.error(f"Ошибка при получении данных по {product_url}: {e}")
        return None, {}

# --- Функция для парсинга одного товара ---
def parse_product(item):
    try:
        # Извлечение данных из элемента поиска
        title = item.query_selector(".div[@data-testid='product-title']").inner_text()
        price = item.query_selector("[data-testid='price']").inner_text()
        link = item.query_selector("a[data-testid='product-link']").get_attribute("href")
        # Получение деталей (из кеша или вызова функции)
        image_url, characteristics = fetch_product_details_cached(link)
        return {
            'title': title,
            'price': price,
            'link': link,
            'image_url': image_url,
            'characteristics': characteristics
        }
    except Exception as e:
        logging.error(f"Ошибка при парсинге товара: {e}")
        return None

# --- Асинхронный сбор поисковых результатов ---
def fetch_search_results(query, max_items=20, max_pages=2):
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        total_fetched = 0
        for page_num in range(1, max_pages + 1):
            url = f"https://ozon.ru/search/?text={query}&page={page_num}"
            try:
                page.goto(url, timeout=60000)
                time.sleep(3)
                items = page.query_selector_all("//div[@data-testid='search-result-item']")
                # Обработка товаров с помощью ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(parse_product, item) for item in items]
                    for future in as_completed(futures):
                        product = future.result()
                        if product:
                            results.append(product)
                            total_fetched += 1
                            if total_fetched >= max_items:
                                break
                if total_fetched >= max_items:
                    break
            except Exception as e:
                logging.error(f"Ошибка при загрузке страницы {url}: {e}")
        browser.close()
    return results

# --- Сохранение данных в базу ---
def save_product_to_db(product, timestamp, query):
    conn = sqlite3.connect('ozon_extended.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO products (timestamp, query, title, price, link, price_num, image_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        query,
        product['title'],
        product['price'],
        product['link'],
        None,
        product['image_url']
    ))
    product_id = c.lastrowid
    for key, value in product['characteristics'].items():
        c.execute('''
            INSERT INTO characteristics (product_id, key, value)
            VALUES (?, ?, ?)
        ''', (product_id, key, value))
    conn.commit()
    conn.close()

# --- Экспорт данных ---
def export_data(format='csv'):
    conn = sqlite3.connect('ozon_extended.db')
    df = pd.read_sql_query("SELECT * FROM products", conn)
    if format == 'csv':
        df.to_csv('ozon_products_export.csv', index=False)
    elif format == 'json':
        df.to_json('ozon_products_export.json', orient='records')
    conn.close()

# --- Streamlit интерфейс ---
st.title("Расширенный сбор данных с OZON (характеристики + изображения)")

search_query = st.text_input("Введите ключевое слово для поиска", value="ноутбук")
if st.button("Запустить сбор данных"):
    with st.spinner("Сбор данных запущен..."):
        results = fetch_search_results(search_query, max_items=20, max_pages=2)
        timestamp = str(datetime.now())
        count = 0
        for product in results:
            save_product_to_db(product, timestamp, search_query)
            count += 1
        st.success(f"Собрано и сохранено товаров: {count}")

# Просмотр последних товаров
st.header("Последние собранные товары")
conn = sqlite3.connect('ozon_extended.db')
df_products = pd.read_sql_query("SELECT * FROM products ORDER BY id DESC LIMIT 20", conn)
conn.close()

for _, row in df_products.iterrows():
    st.subheader(row['title'])
    if row['image_url']:
        st.image(row['image_url'], width=200)
    st.write(f"Цена: {row['price']}")
    st.write(f"[Ссылка на товар]({row['link']})")
    # Характеристики
    conn = sqlite3.connect('ozon_extended.db')
    chars = pd.read_sql_query(f"SELECT key, value FROM characteristics WHERE product_id = {row['id']}", conn)
    conn.close()
    if not chars.empty:
        st.write("Характеристики:")
        for _, ch_row in chars.iterrows():
            st.write(f"**{ch_row['key']}**: {ch_row['value']}")

# Экспорт данных
st.header("Экспорт данных")
if st.button("Экспортировать в CSV"):
    export_data('csv')
    st.success("Данные экспортированы в ozon_products_export.csv")
if st.button("Экспортировать в JSON"):
    export_data('json')
    st.success("Данные экспортированы в ozon_products_export.json")
