import os
import io
import streamlit as st
from PIL import Image
from rembg import remove
import cv2
import numpy as np

def process_image(image_bytes):
    # Удаление фона
    input_image = Image.open(io.BytesIO(image_bytes))
    output_np = remove(np.array(input_image))
    output_image = Image.fromarray(output_np)

    # Обеспечиваем соотношение сторон 3:4
    width, height = output_image.size
    target_ratio = 3 / 4

    current_ratio = width / height
    if current_ratio > target_ratio:
        # Обрезать по ширине
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        output_image = output_image.crop((left, 0, right, height))
    else:
        # Обрезать по высоте
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        output_image = output_image.crop((0, top, width, bottom))
    # Масштабировать к допустимому разрешению (от 200x200 до 4320x7680)
    max_width, max_height = 4320, 7680
    output_image.thumbnail((max_width, max_height), Image.ANTIALIAS)
    return output_image

st.title("Обработка изображений товара")

uploaded_files = st.file_uploader("Загрузите изображения", type=["jpg", "jpeg", "png", "heic", "webp"], accept_multiple_files=True)

save_folder = st.text_input("Папка для сохранения обработанных изображений", value="обработанные_фото")

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image_bytes = uploaded_file.read()
        processed_image = process_image(image_bytes)
        save_path = os.path.join(save_folder, uploaded_file.name)
        processed_image.save(save_path)
        st.image(processed_image, caption=f"Обработано: {uploaded_file.name}")

st.write("Обработка завершена.")
