import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from PIL import Image

st.title("Продвинутая карточка товара для маркетплейса")

# Ввод основных данных
product_name = st.text_input("Название товара")
description = st.text_area("Описание товара")
price = st.number_input("Цена", min_value=0.0, format="%.2f")
category = st.selectbox("Категория", ["Электроника", "Одежда", "Дом и сад", "Книги", "Другое"])
stock = st.number_input("Наличие на складе", min_value=0)
characteristics = st.text_area("Характеристики (по строке)")
images = st.file_uploader("Загрузите изображения", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Предварительный просмотр
if st.button("Показать предварительный просмотр"):
    st.subheader("Предварительный просмотр карточки")
    col1, col2 = st.columns([1, 2])
    with col1:
        for img in images:
            st.image(img, width=150)
    with col2:
        st.markdown(f"## {product_name}")
        st.markdown(f"**Описание:** {description}")
        st.markdown(f"**Категория:** {category}")
        st.markdown(f"**Цена:** ${price:.2f}")
        st.markdown(f"**Наличие на складе:** {stock}")
        st.markdown("### Характеристики")
        for line in characteristics.splitlines():
            st.markdown(f"- {line}")

# Генерация PDF
def create_pdf():
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Заголовок
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, product_name)

    # Изображения
    y_position = height - 150
    for img_file in images:
        img = Image.open(img_file)
        img.thumbnail((150, 150))
        img_io = BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        c.drawInlineImage(img_io, 50, y_position - 150, width=150, height=150)
        y_position -= 160

    # Описание и характеристики
    c.setFont("Helvetica", 12)
    c.drawString(220, height - 100, f"Описание: {description}")
    c.drawString(220, height - 120, f"Категория: {category}")
    c.drawString(220, height - 140, f"Цена: ${price:.2f}")
    c.drawString(220, height - 160, f"Наличие: {stock}")

    c.drawString(50, y_position, "Характеристики:")
    y_char = y_position - 20
    for line in characteristics.splitlines():
        c.drawString(70, y_char, f"- {line}")
        y_char -= 20

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Экспорт в PDF
if st.button("Скачать карточку (PDF)"):
    if not product_name or not description:
        st.error("Пожалуйста, заполните все обязательные поля.")
    else:
        pdf_buffer = create_pdf()
        st.download_button("Скачать PDF", data=pdf_buffer, file_name="card_marketplace.pdf", mime="application/pdf")
