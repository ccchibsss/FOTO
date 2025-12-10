import streamlit as st
import pandas as pd
import time
import random
import undetected_chromedriver as uc
from bs4 import BeautifulSoup

# Функция для получения драйвера
def get_webdriver():
    options = uc.ChromeOptions()
    options.headless = False  # Открыть браузер в обычном режиме
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/115.0.0.0 Safari/537.36")
    driver = uc.Chrome(options=options)
    return driver

# Функция для загрузки страницы
def fetch_selenium(url):
    driver = get_webdriver()
    try:
        driver.get(url)
        # Имитация поведения человека
        time.sleep(random.uniform(3, 6))
        html = driver.page_source
        return html
    except Exception as e:
        st.write(f"Ошибка при загрузке {url}: {e}")
        return None
    finally:
        driver.quit()

# Основной класс парсера
class OzonParser:
    def __init__(self, catalog_url, max_pages=3):
        self.catalog_url = catalog_url
        self.max_pages = max_pages
        self.product_links = []
        self.data = []

    def get_page_links(self, page_url):
        html = fetch_selenium(page_url)
        if not html:
            return []
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.select('a[data-testid="product-card-title"]')  # селектор для ссылок
        hrefs = []
        for link in links:
            href = link.get('href')
            if href and not href.startswith('http'):
                href = 'https://ozon.ru' + href
            hrefs.append(href)
        return hrefs

    def get_product_details(self, url):
        html = fetch_selenium(url)
        if not html:
            return
        soup = BeautifulSoup(html, 'html.parser')
        name_tag = soup.find('h1', {'data-testid': 'product-title'})
        price_tag = soup.find('div', {'data-testid': 'price'})
        image_tag = soup.find('img', {'data-testid': 'product-image'})
        name = name_tag.get_text(strip=True) if name_tag else 'Нет названия'
        price = price_tag.get_text(strip=True) if price_tag else 'Нет цены'
        image = image_tag.get('src') if image_tag else ''
        self.data.append({
            'Название': name,
            'Цена': price,
            'Ссылка': url,
            'Фото': image
        })

    def run(self):
        for page in range(1, self.max_pages + 1):
            page_url = self.catalog_url.replace('{page}', str(page))
            st.write(f'Обработка страницы: {page_url}')
            links = self.get_page_links(page_url)
            self.product_links.extend(links)
            # Можно добавить задержки между страницами
            time.sleep(random.uniform(2, 4))
        st.write(f'Обнаружено товаров: {len(self.product_links)}')
        for url in self.product_links:
            st.write(f'Обработка товара: {url}')
            self.get_product_details(url)
            # Небольшая задержка между товарами
            time.sleep(random.uniform(1, 3))
        return self.data

# Основное приложение Streamlit
def main():
    st.title("Парсер Ozon с обходом защиты")
    st.write("Настраивайте параметры и запускайте.")

    catalog_url = st.text_input(
        "URL каталога (используйте {page} для номера страницы)",
        value='https://ozon.ru/category/tormoznye-diski-38222/lynxauto-86228624/?page={page}'
    )
    max_pages = st.number_input("Максимальное число страниц", min_value=1, max_value=20, value=3)

    if st.button("Запустить парсинг"):
        parser = OzonParser(catalog_url, max_pages)
        with st.spinner('Парсинг идет...'):
            data = parser.run()
        df = pd.DataFrame(data)
        st.success('Парсинг завершен!')
        st.dataframe(df)
        # Сохраняем и даем скачать
        df.to_excel('ozon_data.xlsx', index=False)
        with open('ozon_data.xlsx', 'rb') as f:
            st.download_button("Скачать Excel", f, "ozon_data.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == "__main__":
    main()
