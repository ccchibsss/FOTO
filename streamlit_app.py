# app.py
import streamlit as st
from streamlit_cropper import st_cropper
import os
import numpy as np
from PIL import Image
from rembg import remove
from diffusers import StableDiffusionInpaintPipeline
import torch
import zipfile
from io import BytesIO
from tqdm import tqdm

# Настройки
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

st.set_page_config(page_title="🧹 Удаление фона и водяных знаков", layout="wide")
st.title("🧹 Массовое удаление фона и водяных знаков")
st.markdown("Загрузите изображения — удалим фон, затем водяные знаки с помощью ИИ.")

# Папки
os.makedirs("input", exist_ok=True)
os.makedirs("nobg", exist_ok=True)
os.makedirs("clean", exist_ok=True)

# Очистка папок при перезагрузке (по желанию, или оставить для кэша)
for folder in ["input", "nobg", "clean"]:
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))

# --- Загрузка ---
st.subheader("📤 1. Загрузите изображения")
uploaded_files = st.file_uploader(
    "Поддержка: JPG, PNG. Можно загрузить ZIP с несколькими файлами.",
    type=["jpg", "jpeg", "png", "zip"],
    accept_multiple_files=False  # Загружаем один архив или несколько файлов
)

image_paths = []

if uploaded_files:
    with st.spinner("Обработка загрузки..."):
        if uploaded_files.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_files, "r") as z:
                z.extractall("input")
            image_paths = [f for f in os.listdir("input") if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        else:
            with open(os.path.join("input", uploaded_files.name), "wb") as f:
                f.write(uploaded_files.getbuffer())
            image_paths = [uploaded_files.name]

    st.success(f"✅ Загружено {len(image_paths)} изображений.")

# --- Удаление фона ---
if image_paths and st.button("🚀 Удалить фон (все изображения)"):
    with st.spinner("Удаляем фон... Это может занять время."):
        for filename in image_paths:
            input_path = os.path.join("input", filename)
            output_path = os.path.join("nobg", f"{os.path.splitext(filename)[0]}.png")
            try:
                img = Image.open(input_path)
                img_no_bg = remove(img)
                img_no_bg.save(output_path, "PNG")
            except Exception as e:
                st.warning(f"Ошибка при обработке {filename}: {e}")
    st.success("✅ Фон удалён со всех изображений!")

# --- Показать пример ---
if os.listdir("nobg"):
    st.subheader("🖼️ 2. Выберите область водяного знака")
    preview_file = st.selectbox("Выберите изображение для настройки", os.listdir("nobg"))
    preview_path = os.path.join("nobg", preview_file)
    preview_img = Image.open(preview_path)

    # Масштабирование для удобства
    max_size = 600
    if max(preview_img.size) > max_size:
        scale = max_size / max(preview_img.size)
        new_size = (int(preview_img.width * scale), int(preview_img.height * scale))
        preview_img = preview_img.resize(new_size, Image.LANCZOS)

    # Интерактивное выделение
    cropped_img = st_cropper(preview_img, realtime_update=True, box_color="#FF0004", aspect_ratio=None)
    st.write("Нарисуйте прямоугольник вокруг водяного знака.")

    # Конвертация выделения в маску
    if cropped_img:
        left, top, right, bottom = st.session_state.get("box", (0, 0, 100, 100))[:4]
        mask = Image.new("L", preview_img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([left, top, right, bottom], fill=255)
        st.image(mask, caption="Маска для удаления")

# --- Удалить водяные знаки ---
if os.listdir("nobg") and st.button("🧹 Удалить водяные знаки (все изображения)"):
    prompt = st.text_input("Промпт для ИИ (описание фона)", "natural background, clean, no text, high quality")
    with st.spinner("Загружаем модель Stable Diffusion Inpainting..."):
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=DTYPE
            ).to(DEVICE)
            st.info("Модель загружена. Начинаем обработку...")
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            st.stop()

    # Прогресс
    progress_bar = st.progress(0)
    for i, filename in enumerate(os.listdir("nobg")):
        if not filename.lower().endswith(".png"):
            continue
        try:
            img_path = os.path.join("nobg", filename)
            img = Image.open(img_path).convert("RGB")

            # Масштабирование (модель лучше работает до 512x512)
            orig_size = img.size
            img_resized = img.resize((512, 512), Image.LANCZOS) if max(img.size) > 512 else img

            # Используем маску из cropper (или дефолтную)
            mask_img = Image.new("L", img_resized.size, 0)
            draw = ImageDraw.Draw(mask_img)
            if 'box' in st.session_state:
                x0, y0, x1, y1 = [int(v * 512 / orig_size[0]) for v in st.session_state.box[:4]]
                draw.rectangle([x0, y0, x1, y1], fill=255)
            else:
                # Если нет маски — автоматическая (низ)
                w, h = img_resized.size
                draw.rectangle([w // 2 - 100, h - 80, w // 2 + 100, h - 20], fill=255)

            # Инпейнтинг
            result = pipe(
                prompt=prompt,
                image=img_resized,
                mask_image=mask_img,
                strength=0.75,
                guidance_scale=7.5,
                num_inference_steps=30
            ).images[0]

            # Восстановить оригинальный размер
            result = result.resize(orig_size, Image.LANCZOS)
            clean_path = os.path.join("clean", filename)
            result.save(clean_path, "PNG")

        except Exception as e:
            st.warning(f"Ошибка при обработке {filename}: {e}")

        progress_bar.progress((i + 1) / len(os.listdir("nobg")))

    st.success("✅ Все водяные знаки удалены!")
    st.balloons()

    # --- Скачивание результата ---
    if os.listdir("clean"):
        st.subheader("✅ Результаты готовы")
        # Показать первое изображение
        result_img = Image.open(os.path.join("clean", os.listdir("clean")[0]))
        st.image(result_img, caption="Обработанное изображение", use_column_width=True)

        # Создание ZIP
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for fname in os.listdir("clean"):
                zf.write(os.path.join("clean", fname), fname)
        zip_buffer.seek(0)

        st.download_button(
            label="📦 Скачать все изображения (ZIP)",
            data=zip_buffer,
            file_name="cleaned_images.zip",
            mime="application/zip"
        )

# --- Подвал ---
st.markdown("---")
st.caption("Создано с ❤️ с использованием Streamlit, rembg и Stable Diffusion.")
