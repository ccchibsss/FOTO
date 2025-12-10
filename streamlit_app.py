import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Frame
from io import BytesIO
from PIL import Image
import requests

# Предположим, есть API маркетплейса (пример)
API_ENDPOINT = "https://api.marketplace.com/products"

st.title("Создание карточки товара с расширенным дизайном и API")

# Ввод данных
product_name = st.text_input("Название товара")
description = st.text_area("Описание товара")
price = st.number_input("Цена", min_value=0.0, format="%.2f")
category = st.selectbox("Категория", ["Электроника", "Одежда", "Дом и сад", "Книги", "Другое"])
stock = st.number_input("Наличие на складе", min_value=0)
characteristics = st.text_area("Характеристики (по строке)")

# Загрузка изображений
images_files = st.file_uploader("Загрузите изображения", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Имитация API-запроса для получения категорий и характеристик
def fetch_category_info(category_name):
    # В реальности тут будет запрос к API
    return {
        "id": 123,
        "name": category_name,
        "description": f"Описание категории {category_name}"
    }

category_info = fetch_category_info(category)

# Предварительный просмотр
if st.button("Показать предварительный просмотр"):
    st.subheader("Предварительный просмотр карточки")
    col1, col2 = st.columns([1, 2])
    with col1:
        for img in images_files:
            st.image(img, width=150)
    with col2:
        st.markdown(f"## {product_name}")
        st.markdown(f"**Описание:** {description}")
        st.markdown(f"**Категория:** {category}")
        st.markdown(f"**Цена:** ${price:.2f}")
        st.markdown(f"**Наличие:** {stock}")
        st.markdown("### Характеристики")
        for line in characteristics.splitlines():
            st.markdown(f"- {line}")

# Создаем более сложный дизайн PDF
def create_advanced_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Фон шапки с градиентом (имитация)
    c.setFillColorRGB(0.2, 0.4, 0.6)
    c.rect(0, height - 100, width, 100, fill=1)

    # Логотип или логотип маркетплейса
    # Можно вставить изображение логотипа
    # c.drawInlineImage("logo.png", 20, height - 80, width=60, height=60)

    # Название товара большими буквами с тенью
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(50, height - 50, product_name)

    # Оформление блока с изображениями (с рамкой)
    y_pos = height - 150
    x_offset = 50
    image_size = 120
    for img in images_files:
        pil_img = Image.open(img).convert("RGBA")
        pil_img.thumbnail((image_size, image_size))
        img_bytes = BytesIO()
        pil_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        c.drawInlineImage(img_bytes, x_offset, y_pos - image_size, width=image_size, height=image_size)
        x_offset += image_size + 20
        if x_offset + image_size > width - 50:
            x_offset = 50
            y_pos -= image_size + 20

    # Описание и характеристики в выделенном блоке
    styles = getSampleStyleSheet()
    styleN = styles['BodyText']
    styleH = styles['Heading2']

    # Создаем параграфы
    description_para = Paragraph(f"<b>Описание:</b> {description}", styleN)
    characteristics_para = Paragraph("<b>Характеристики:</b><br/>" + "<br/>".join([f"- {line}" for line in characteristics.splitlines()]), styleN)

    # Рамки для текста
    frame_desc = Frame(50, y_pos - 200, 500, 150, showBoundary=1)
    frame_desc.addFromList([description_para], c)

    frame_char = Frame(50, y_pos - 400, 500, 180, showBoundary=1)
    frame_char.addFromList([characteristics_para], c)

    # Цена и категория
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos - 420, f"Цена: ${price:.2f}")
    c.drawString(50, y_pos - 440, f"Категория: {category}")

    # Можно добавить дополнительные элементы по дизайну

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# API-запрос для публикации товара
def publish_to_marketplace(data):
    # В реальности делается POST-запрос с auth и payload
    response = requests.post(API_ENDPOINT, json=data)
    if response.status_code == 201:
        return True
    return False

# Кнопка публикации
if st.button("Опубликовать на маркетплейсе"):
    # Тут подготовим данные
    product_data = {
        "name": product_name,
        "description": description,
        "price": price,
        "category_id": category_info["id"],
        "stock": stock,
        "images": [],  # Можно прикреплять URL или base64
    }
    # Загруженные изображения можно сохранять или конвертировать
    # Например, в base64
    for img in images_files:
        img_bytes = img.read()
        encoded = base64.b64encode(img_bytes).decode()
        product_data["images"].append(encoded)

    success = publish_to_marketplace(product_data)
    if success:
        st.success("Товар успешно опубликован!")
    else:
        st.error("Ошибка публикации. Попробуйте позже.")

# Скачать карточку
if st.button("Скачать расширенную карточку (PDF)"):
    pdf_buffer = create_advanced_pdf()
    st.download_button("Скачать PDF", data=pdf_buffer, file_name="advanced_product_card.pdf", mime="application/pdf")
