import streamlit as st
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import random

# Заголовки для HTTP-запросов, маскирующие ботов
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}
# Список прокси-серверов (может быть пустым, если прокси не используются)
PROXIES = []

# Асинхронная функция для отправки GET-запроса с использованием aiohttp
async def fetch(session, url):
    # Выбираем случайный прокси, если есть
    proxy = random.choice(PROXIES) if PROXIES else None
    try:
        # Отправляем запрос
        async with session.get(url, headers=HEADERS, proxy=proxy, timeout=15) as response:
            response.raise_for_status()  # Проверка статуса
            return await response.text()  # Возвращаем HTML содержимое
    except Exception as e:
        # В случае ошибки выводим сообщение
        st.write(f"Ошибка при запросе {url}: {e}")
        return None

# Основной класс парсера маркетплейса
class MarketplaceScraper:
    def __init__(self, config):
        self.config = config  # Конфигурация парсера, параметры сайта
        self.product_links = []  # Список ссылок на товары
        self.data = []  # Собранные данные о товарах

    # Получение ссылок на товары со страницы каталога
    async def get_page_links(self, session, page_url):
        html = await fetch(session, page_url)  # Получаем HTML страницы
        if not html:
            return []  # Если ошибка, возвращаем пустой список
        soup = BeautifulSoup(html, 'html.parser')  # Парсим HTML
        links = soup.select(self.config['product_link_selector'])  # Ищем ссылки по селектору
        hrefs = []
        for link in links:
            href = link.get('href')  # Получаем href
            if href and not href.startswith('http'):
                # Если относительная ссылка, добавляем базовый URL
                base = self.config['base_url']
                if not base.endswith('/'):
                    base += '/'
                href = base + href.lstrip('/')
            hrefs.append(href)
        return hrefs

    # Получение данных о товаре по ссылке
    async def get_product_details(self, session, url):
        html = await fetch(session, url)  # Загружаем страницу товара
        if not html:
            return
        soup = BeautifulSoup(html, 'html.parser')
        # Находим название товара
        name_tag = soup.select_one(self.config['product_name_selector'])
        # Находим цену
        price_tag = soup.select_one(self.config['product_price_selector'])
        # Находим изображение
        image_tag = soup.select_one(self.config['product_image_selector'])
        # Извлекаем текст или устанавливаем дефолт
        name = name_tag.get_text(strip=True) if name_tag else 'Нет названия'
        price = price_tag.get_text(strip=True) if price_tag else 'Нет цены'
        image = image_tag.get('src') if image_tag else ''
        # Обработка относительных путей к изображению
        if image and not image.startswith('http'):
            base = self.config['base_url']
            if not base.endswith('/'):
                base += '/'
            image = base + image.lstrip('/')
        # Добавляем данные в список
        self.data.append({
            'Название': name,
            'Цена': price,
            'Ссылка': url,
            'Фото': image
        })

    # Основной метод запуска парсинга
    async def run(self):
        # Создаем асинхронную сессию
        async with aiohttp.ClientSession() as session:
            tasks_pages = []
            # Создаем задачи для загрузки страниц каталога
            for page in range(1, self.config['max_pages'] + 1):
                page_url = self.config['catalog_url'].replace('{page}', str(page))
                tasks_pages.append(self.get_page_links(session, page_url))
            # Запускаем задачи и собираем ссылки
            pages_links = await asyncio.gather(*tasks_pages)
            for links in pages_links:
                self.product_links.extend(links)
            # Выводим количество найденных товаров
            st.write(f'Найдено товаров: {len(self.product_links)}')
            # Создаем задачи для получения данных о каждом товаре
            tasks_products = []
            for url in self.product_links:
                tasks_products.append(self.get_product_details(session, url))
            # Запускаем и ждем завершения
            await asyncio.gather(*tasks_products)
        return self.data  # Возвращаем собранные данные

# Основная функция запуска приложения
def main():
    st.title("Гибкий и обход блокировок парсер маркетплейсов")
    st.write("Настраивайте параметры для обхода блокировок и поиска данных.")

    # Ввод конфигурационных данных
    catalog_url = st.text_input(
        "URL каталога (используйте {page} для пагинации)", 
        value='https://пример-маркетплейса.com/каталог?page={page}'
    )
    base_url = st.text_input(
        "Базовый URL сайта (например, https://пример-маркетплейса.com)", 
        value='https://пример-маркетплейса.com'
    )
    product_link_selector = st.text_input(
        "CSS-селектор ссылок на товары", 
        value='a.product-link'
    )
    product_name_selector = st.text_input(
        "CSS-селектор названия товара", 
        value='h1.product-title'
    )
    product_price_selector = st.text_input(
        "CSS-селектор цены", 
        value='span.price'
    )
    product_image_selector = st.text_input(
        "CSS-селектор изображения", 
        value='img.product-image'
    )
    max_pages = st.number_input(
        "Максимальное число страниц", 
        min_value=1, max_value=50, value=5
    )

    # Настройка прокси
    st.write("Настройка прокси (оставьте пустым для отключения):")
    PROXIES_INPUT = st.text_area("Прокси (по одному на строку):", value='').splitlines()
    # Обновляем глобальный список PROXIES
    PROXIES.clear()
    PROXIES.extend([p.strip() for p in PROXIES_INPUT if p.strip()])

    # Запуск парсинга по кнопке
    if st.button("Запустить парсинг"):
        # Формируем конфигурацию
        config = {
            'catalog_url': catalog_url,
            'base_url': base_url,
            'product_link_selector': product_link_selector,
            'product_name_selector': product_name_selector,
            'product_price_selector': product_price_selector,
            'product_image_selector': product_image_selector,
            'max_pages': max_pages
        }
        # Создаем объект парсера
        scraper = MarketplaceScraper(config)
        # Запускаем асинхронную функцию
        with st.spinner('Парсинг идет...'):
            data = asyncio.run(scraper.run())
        # Сообщение о завершении
        st.success('Парсинг завершен!')
        # Создаем DataFrame
        df = pd.DataFrame(data)
        # Отображаем таблицу
        st.dataframe(df)

        # Сохраняем в Excel файл
        with pd.ExcelWriter('marketplace_data.xlsx', engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        # Загружаем файл для скачивания
        with open('marketplace_data.xlsx', 'rb') as f:
            data_bytes = f.read()
        st.download_button(
            "Скачать Excel",
            data=data_bytes,
            file_name='marketplace_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

if __name__ == "__main__":
    main()
