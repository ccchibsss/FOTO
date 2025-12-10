# Название папки проекта
$projectFolder = "marketplace_parser_auto"

# Создаем папку проекта
if (!(Test-Path $projectFolder)) {
    New-Item -ItemType Directory -Path $projectFolder | Out-Null
}
Set-Location $projectFolder

# Создаем виртуальное окружение
python -m venv venv

# Активируем виртуальное окружение
$activateScript = ".\venv\Scripts\Activate.ps1"
if (!(Test-Path $activateScript)) {
    Write-Error "Не удалось найти скрипт активации виртуального окружения."
    exit
}
& $activateScript

# Обновляем pip
pip install --upgrade pip

# Устанавливаем необходимые библиотеки
pip install streamlit beautifulsoup4 pandas aiohttp

# Создаем основной скрипт парсера
$scriptContent = @"
import streamlit as st
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import random

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                  ' Chrome/115.0.0.0 Safari/537.36'
}
PROXIES = []

async def fetch(session, url):
    proxy = random.choice(PROXIES) if PROXIES else None
    try:
        async with session.get(url, headers=HEADERS, proxy=proxy, timeout=15) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        st.write(f"Ошибка при запросе {url}: {e}")
        return None

class MarketplaceScraper:
    def __init__(self, config):
        self.config = config
        self.product_links = []
        self.data = []

    async def get_page_links(self, session, page_url):
        html = await fetch(session, page_url)
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

    async def get_product_details(self, session, url):
        html = await fetch(session, url)
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

    async def run(self):
        async with aiohttp.ClientSession() as session:
            tasks_pages = []
            for page in range(1, self.config['max_pages'] + 1):
                page_url = self.config['catalog_url'].replace('{page}', str(page))
                tasks_pages.append(self.get_page_links(session, page_url))
            pages_links = await asyncio.gather(*tasks_pages)
            for links in pages_links:
                self.product_links.extend(links)
            st.write(f'Найдено товаров: {len(self.product_links)}')
            tasks_products = []
            for url in self.product_links:
                tasks_products.append(self.get_product_details(session, url))
            await asyncio.gather(*tasks_products)
        return self.data

st.title("Гибкий и обход блокировок парсер маркетплейсов")
st.write("Настраивайте параметры для обхода блокировок и поиска данных.")

catalog_url = st.text_input("URL каталога (используйте {page} для пагинации)", value='https://пример-маркетплейса.com/каталог?page={page}')
base_url = st.text_input("Базовый URL сайта (например, https://пример-маркетплейса.com)", value='https://пример-маркетплейса.com')
product_link_selector = st.text_input("CSS-селектор ссылок на товары", value='a.product-link')
product_name_selector = st.text_input("CSS-селектор названия товара", value='h1.product-title')
product_price_selector = st.text_input("CSS-селектор цены", value='span.price')
product_image_selector = st.text_input("CSS-селектор изображения", value='img.product-image')
max_pages = st.number_input("Максимальное число страниц", min_value=1, max_value=50, value=5)

st.write("Настройка прокси (оставьте пустым для отключения):")
PROXIES_INPUT = st.text_area("Прокси (по одному на строку):", value='').splitlines()
PROXIES = [p.strip() for p in PROXIES_INPUT if p.strip()]

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
        data = asyncio.run(scraper.run())
    st.success('Парсинг завершен!')
    df = pd.DataFrame(data)
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Скачать CSV", data=csv, file_name='marketplace_data.csv', mime='text/csv')
"@

# Записываем скрипт в файл
$scriptPath = ".\marketplace_parser.py"
$scriptContent | Out-File -FilePath $scriptPath -Encoding utf8

# Запускаем приложение Streamlit
streamlit run $scriptPath
