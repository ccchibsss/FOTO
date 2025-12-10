import os
import io
import sys
import numpy as np
import cv2
from PIL import Image, ImageOps
try:
    import streamlit as st
except Exception:
    print("Streamlit не установлен. Установите: pip install streamlit")
    sys.exit(1)

# optional HEIC
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# optional background removal
try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None

# optional TF model for watermark segmentation
try:
    import tensorflow as tf
    from tensorflow.keras import models
except Exception:
    tf = None

st.set_page_config(page_title="Batch BG & Watermark Remover", layout="wide")
st.title("Массовое удаление фона и водяных знаков")

# Sidebar settings
st.sidebar.header("Настройки")
save_folder = st.sidebar.text_input("Папка для сохранения", value="processed_photos")
os.makedirs(save_folder, exist_ok=True)

format_choice = st.sidebar.selectbox("Формат сохранения", ["Оставить оригинал", "PNG", "JPEG", "WEBP"])
quality = st.sidebar.slider("Качество JPEG/WEBP", 50, 100, 95)
bg_method = st.sidebar.selectbox("Метод удаления фона", ["rembg (если доступен)", "grabcut (OpenCV)"])
use_segmentation = st.sidebar.checkbox("Использовать модель сегментации водяных знаков (.h5)", value=False)
model_path = st.sidebar.text_input("Путь к модели (если включено)", value="watermark_segmentation_model.h5")
st.sidebar.markdown("Ограничения: min 200x200, max 4320x7680. Итоговое соотношение: 3:4")

uploaded_files = st.file_uploader("Загрузите изображения (jpg, png, webp, heic и т.д.)", accept_multiple_files=True)

# Helpers
def load_segmentation_model(path):
    if not use_segmentation or tf is None:
        return None
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            st.sidebar.error(f"Не удалось загрузить модель: {e}")
    else:
        st.sidebar.warning("Файл модели не найден по указанному пути.")
    return None

def remove_background_rembg(pil_img):
    try:
        arr = np.array(pil_img.convert("RGBA"))
        out = rembg_remove(arr)
        return Image.fromarray(out).convert("RGBA")
    except Exception:
        return pil_img.convert("RGBA")

def remove_background_grabcut(pil_img, iter_count=5):
    img = np.array(pil_img.convert("RGB"))[:, :, ::-1].copy()  # BGR
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return pil_img.convert("RGBA")
    mask = np.zeros((h, w), np.uint8)
    rect = (max(1, w//20), max(1, h//20), max(2, w - w//10), max(2, h - h//10))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        alpha = (mask2 * 255).astype('uint8')
        b, g, r = cv2.split(img)
        rgba = cv2.merge([r, g, b, alpha])
        return Image.fromarray(rgba, mode="RGBA")
    except Exception:
        return pil_img.convert("RGBA")

def crop_to_ratio(pil_img, ratio=3/4):
    w, h = pil_img.size
    current = w / h
    if abs(current - ratio) < 1e-6:
        return pil_img
    if current > ratio:
        new_w = int(h * ratio)
        left = (w - new_w) // 2
        return pil_img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / ratio)
        top = (h - new_h) // 2
        return pil_img.crop((0, top, w, top + new_h))

def ensure_resolution(pil_img, min_size=(200,200), max_size=(4320,7680)):
    w, h = pil_img.size
    max_w, max_h = max_size
    if w > max_w or h > max_h:
        pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
        w, h = pil_img.size
    min_w, min_h = min_size
    if w < min_w or h < min_h:
        scale = max(min_w / w, min_h / h)
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return pil_img

def save_image_preserve_name(pil_img, original_name, out_dir, fmt_choice):
    name, ext = os.path.splitext(original_name)
    ext = ext.lower().lstrip('.')
    if fmt_choice == "Оставить оригинал":
        out_ext = ext if ext else "png"
        out_name = f"{name}.{out_ext}"
    else:
        out_ext = fmt_choice.lower()
        out_name = f"{name}.{('jpg' if out_ext=='jpeg' else out_ext)}"
    out_path = os.path.join(out_dir, out_name)
    save_kwargs = {}
    if out_ext in ("jpg", "jpeg"):
        if "A" in pil_img.mode:
            bg = Image.new("RGB", pil_img.size, (255,255,255))
            bg.paste(pil_img, mask=pil_img.split()[-1])
            to_save = bg
        else:
            to_save = pil_img.convert("RGB")
        save_kwargs["quality"] = quality
        to_save.save(out_path, format="JPEG", **save_kwargs)
    elif out_ext == "webp":
        save_kwargs["quality"] = quality
        pil_img.save(out_path, format="WEBP", **save_kwargs)
    elif out_ext == "png":
        pil_img.save(out_path, format="PNG")
    else:
        # fallback
        pil_img.save(out_path)
    return out_path

def remove_watermark_model(cv_bgr, model):
    try:
        h, w = cv_bgr.shape[:2]
        inp = cv2.resize(cv_bgr, (256,256)) / 255.0
        inp = np.expand_dims(inp, axis=0)
        pred = model.predict(inp, verbose=0)[0]
        mask = (pred[...,0] > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (w,h))
        # invert mask: watermark area -> inpaint
        to_inpaint_mask = (mask > 0).astype(np.uint8) * 255
        inpainted = cv2.inpaint(cv_bgr, to_inpaint_mask, 3, cv2.INPAINT_TELEA)
        return inpainted
    except Exception:
        return cv_bgr

def remove_watermark_heuristic(cv_bgr):
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # Detect bright low-saturation areas (typical white watermarks)
    hsv = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    s = hsv[:,:,1]
    mask1 = ((v > 200) & (s < 100)).astype(np.uint8) * 255
    # edge-based detection (text edges)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.bitwise_or(mask1, edges)
    # focus near borders (watermarks often in corners/bottom)
    border_mask = np.zeros_like(mask)
    bw = max(20, w//5); bh = max(20, h//5)
    border_mask[0:bh, 0:w] = 255
    border_mask[h-bh:h, 0:w] = 255
    border_mask[0:h, 0:bw] = 255
    border_mask[0:h, w-bw:w] = 255
    mask = cv2.bitwise_and(mask, border_mask)
    # clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if np.count_nonzero(mask) < 10:
        return cv_bgr
    try:
        inpainted = cv2.inpaint(cv_bgr, mask, 3, cv2.INPAINT_TELEA)
        return inpainted
    except Exception:
        return cv_bgr

# Load segmentation model if requested
model = None
if use_segmentation:
    model = load_segmentation_model(model_path)

# Processing
if uploaded_files:
    total = len(uploaded_files)
    progress = st.progress(0)
    processed = 0
    for uploaded in uploaded_files:
        try:
            data = uploaded.read()
            pil = Image.open(io.BytesIO(data))
            pil = pil.convert("RGBA")

            # 1) remove background
            if bg_method == "rembg (если доступен)" and rembg_remove is not None:
                pil_nobg = remove_background_rembg(pil)
            else:
                pil_nobg = remove_background_grabcut(pil)

            # 2) remove watermark
            cv_img = cv2.cvtColor(np.array(pil_nobg), cv2.COLOR_RGBA2BGR)
            if model is not None:
                cv_processed = remove_watermark_model(cv_img, model)
            else:
                cv_processed = remove_watermark_heuristic(cv_img)

            pil_after = Image.fromarray(cv2.cvtColor(cv_processed, cv2.COLOR_BGR2RGBA))

            # 3) crop to 3:4 and ensure size constraints
            pil_after = crop_to_ratio(pil_after, ratio=3/4)
            pil_after = ensure_resolution(pil_after, min_size=(200,200), max_size=(4320,7680))

            # 4) save
            out_path = save_image_preserve_name(pil_after, uploaded.name, save_folder, format_choice)
            st.image(pil_after, caption=f"Сохранено: {os.path.basename(out_path)}", use_column_width=False)
        except Exception as e:
            st.error(f"Ошибка при обработке {uploaded.name}: {e}")
        processed += 1
        progress.progress(int(processed/total*100))
    st.success(f"Готово: {processed}/{total}. Файлы сохранены в: {os.path.abspath(save_folder)}")
else:
    st.info("Загрузите файлы для массовой обработки.")
