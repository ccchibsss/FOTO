import streamlit as st
import pandas as pd
import time
import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Настройка Selenium WebDriver
def get_webdriver():
    options = Options()
    options.headless = True  # Без GUI
    options.add_argument("--disable-blink-features=AutomationControlled")
    # Можно добавить еще опции для маскировки
    # options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")
    # Укажите путь к chromedriver, если он не в PATH
    service = Service()  # или Service('/path/to/chromedriver')
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# Функция для загрузки страницы через Selenium
def fetch_selenium(url):
    driver = get_webdriver()
    try:
        # Задержка, чтобы сайт успел прогрузиться
        driver.get(url)
        time.sleep(random.uniform(2, 4))
        html = driver.page_source
        return html
    except Exception as e:
        st.write(f"Ошибка при загрузке {url}: {e}")
        return None
    finally:
        driver.quit()

# Основной класс
class MarketplaceScraper:
    def __init__(self, config):
        self.config = config
        self.product_links = []
        self.data = []

    def get_page_links(self, page_url):
        html = fetch_selenium(page_url)
        if not html:
            return []
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.select(self.config['product_link_selector'])
        hrefs = []
        for link in links:
            href = link.get('href')
            if href and not href.startswith('http'):
                base = self.config['base_url']
                if not base.endswith('/'):
                    base += '/'
                href = base + href.lstrip('/')
            hrefs.append(href)
        return hrefs

    def get_product_details(self, url):
        html = fetch_selenium(url)
        if not html:
            return
        soup = BeautifulSoup(html, 'html.parser')
        name_tag = soup.select_one(self.config['product_name_selector'])
        price_tag = soup.select_one(self.config['product_price_selector'])
        image_tag = soup.select_one(self.config['product_image_selector'])
        name = name_tag.get_text(strip=True) if name_tag else 'Нет названия'
        price = price_tag.get_text(strip=True) if price_tag else 'Нет цены'
        image = image_tag.get('src') if image_tag else ''
        if image and not image.startswith('http'):
            base = self.config['base_url']
            if not base.endswith('/'):
                base += '/'
            image = base + image.lstrip('/')
        self.data.append({
            'Название': name,
            'Цена': price,
            'Ссылка': url,
            'Фото': image
        })

    def run(self):
        for page in range(1, self.config['max_pages'] + 1):
            page_url = self.config['catalog_url'].replace('{page}', str(page))
            links = self.get_page_links(page_url)
            self.product_links.extend(links)
        st.write(f'Найдено товаров: {len(self.product_links)}')
        for url in self.product_links:
            self.get_product_details(url)
        return self.data

# Основная функция
def main():
    st.title("Парсер маркетплейсов с обходом защиты (Selenium)")
    st.write("Настраивайте параметры и запускайте.")

    catalog_url = st.text_input("URL каталога (используйте {page})", value='https://пример-маркетплейса.com/каталог?page={page}')
    base_url = st.text_input("Базовый URL сайта", value='https://пример-маркетплейса.com')
    product_link_selector = st.text_input("CSS-селектор ссылок на товары", value='a.product-link')
    product_name_selector = st.text_input("CSS-селектор названия товара", value='h1.product-title')
    product_price_selector = st.text_input("CSS-селектор цены", value='span.price')
    product_image_selector = st.text_input("CSS-селектор изображения", value='img.product-image')
    max_pages = st.number_input("Максимальное число страниц", min_value=1, max_value=50, value=5)

    if st.button("Запустить парсинг"):
        config = {
            'catalog_url': catalog_url,
            'base_url': base_url,
            'product_link_selector': product_link_selector,
            'product_name_selector': product_name_selector,
            'product_price_selector': product_price_selector,
            'product_image_selector': product_image_selector,
            'max_pages': max_pages
        }
        scraper = MarketplaceScraper(config)
        with st.spinner('Парсинг идет...'):
            data = scraper.run()
        st.success('Парсинг завершен!')
        df = pd.DataFrame(data)
        st.dataframe(df)
        # Сохраняем и скачиваем
        df.to_excel('marketplace_data.xlsx', index=False)
        with open('marketplace_data.xlsx', 'rb') as f:
            st.download_button("Скачать Excel", f, "marketplace_data.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == "__main__":
    main()
