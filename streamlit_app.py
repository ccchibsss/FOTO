# Проверка и установка distutils, если его нет
try:
    import distutils
except ModuleNotFoundError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
    import distutils

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFilter
import io
import tempfile
from moviepy.editor import VideoFileClip, ImageSequenceClip
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch

# Загрузка моделей с настройками порогов
@st.cache(allow_output_mutation=True)
def load_mask_rcnn_model(model_name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # по умолчанию, переопределяется порогом
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    return predictor

# Поддерживаемые модели
model_options = {
    "Mask R-CNN ResNet50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "Mask R-CNN ResNet101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
}

# Настройки порогов для каждой модели (по умолчанию)
model_thresholds = {
    "Mask R-CNN ResNet50": 0.5,
    "Mask R-CNN ResNet101": 0.5
}

# Функция обнаружения масок
def detect_masks(image_cv2, predictor, threshold):
    outputs = predictor(image_cv2)
    masks = outputs["instances"].pred_masks.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    selected_masks = [m for m, s in zip(masks, scores) if s >= threshold]
    selected_scores = [s for s in scores if s >= threshold]
    return selected_masks, selected_scores

# Более сложная автоматическая логика
def auto_mode_complex(masks, scores, image_shape):
    if not masks:
        return "Цвет"
    
    total_mask_area = sum(np.sum(mask) for mask in masks)
    mask_density = total_mask_area / (image_shape[0] * image_shape[1])
    avg_score = np.mean(scores)
    max_mask_size = max(np.sum(mask) for mask in masks)
    normalized_mask_size = max_mask_size / (image_shape[0] * image_shape[1])
    # Распределение масок по изображению
    mask_positions = [np.where(mask) for mask in masks]
    # Можно добавить дополнительные правила по расположению, если нужно

    # Правила:
    if mask_density > 0.2:
        return "Размытие"
    elif normalized_mask_size < 0.01 and avg_score < 0.5:
        return "Цвет"
    elif mask_density > 0.1 and normalized_mask_size > 0.05:
        return "Размытие"
    else:
        return "Цвет"

def remove_watermarks(image, masks, background_img=None, mode="Цвет", color="#000000", blur_radius=5):
    image_np = np.array(image)
    if background_img is not None:
        background = np.array(background_img.resize(image.size))
    else:
        if mode == "Цвет":
            fill_color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            background = np.full_like(image_np, fill_color_rgb)
        elif mode == "Размытие":
            bg_image = Image.fromarray(image_np)
            blurred = bg_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            background = np.array(blurred)
        else:
            mean_color = np.mean(image_np, axis=(0, 1)).astype(np.uint8)
            background = np.full_like(image_np, mean_color)
    for mask in masks:
        mask_bool = mask.astype(bool)
        image_np[mask_bool] = background[mask_bool]
    return Image.fromarray(image_np)

def process_image(image_bytes, predictor, thresholds, background_img, replace_mode, fill_color, blur_radius):
    image_cv2 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    selected_model_name = st.session_state['model_name']
    threshold = thresholds[selected_model_name]
    masks, scores = detect_masks(image_cv2, predictor, threshold)
    image_size = image_cv2.shape[:2]
    # Используем более сложную автоматическую логику
    auto_mode_choice = auto_mode_complex(masks, scores, image_size)
    mode = auto_mode_choice if replace_mode == "Авто" else replace_mode
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Предварительный просмотр маски
    if masks:
        mask_overlay = np.zeros_like(np.array(image))
        for mask in masks:
            mask_overlay[mask.astype(bool)] = [255, 0, 0]
        overlay_img = Image.fromarray(mask_overlay).convert("RGBA")
        preview = image.convert("RGBA")
        combined = Image.alpha_composite(preview, overlay_img)
        st.image(combined, caption="Предварительный просмотр маски (красный — водяной знак)")
        # Подтверждение для удаления
        if st.button("Удалить водяной знак с этим изображением?"):
            image = remove_watermarks(image, masks, background_img, mode, fill_color, blur_radius)
    else:
        st.info("Водяные знаки не обнаружены.")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

def process_video(video_path, predictor, thresholds, background_img, replace_mode, fill_color, blur_radius):
    clip = VideoFileClip(video_path)
    processed_frames = []

    for frame in clip.iter_frames():
        frame_bytes = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes()
        processed_frame = process_image(frame_bytes, predictor, thresholds, background_img, replace_mode, fill_color, blur_radius)
        processed_frames.append(np.array(processed_frame))
    processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    processed_clip.write_videofile(output_path, codec="libx264", logger=None)
    return output_path

def main():
    st.title("🖼️✨ Расширенное автоматическое удаление водяных знаков")
    st.markdown("## 🚀 Выбор модели обнаружения")
    model_name = st.selectbox("Модель обнаружения водяных знаков", list(model_options.keys()))
    st.session_state['model_name'] = model_name
    predictor = load_mask_rcnn_model(model_options[model_name])

    # Настройки порогов для каждой модели
    st.markdown("## 🔧 Настройки порогов для моделей")
    for m in model_options:
        default_threshold = model_thresholds.get(m, 0.5)
        threshold_input = st.slider(f"Порог для {m}", 0.0, 1.0, default_threshold, 0.05)
        model_thresholds[m] = threshold_input

    st.markdown("## ⚙️ Настройки обработки")
    replace_mode = st.selectbox("Режим замещения", ["Цвет", "Размытие", "Авто"])
    fill_color = st.color_picker("Цвет заливки", "#000000")
    blur_radius = st.slider("Радиус размытия", 1, 25, 5)

    st.markdown("## 🖼️ Выбор фона")
    background_file = st.file_uploader("Загрузите изображение фона", type=["png", "jpg", "jpeg"])
    background_img = None
    if background_file:
        background_img = Image.open(background_file).convert("RGB")
        st.image(background_img, caption="Выбранный фон", width=300)

    st.markdown("## 📂 Загрузка файлов")
    uploaded_files = st.file_uploader("Загрузите изображения или видео", type=["png", "jpg", "jpeg", "mp4", "avi"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Обработка файла: {uploaded_file.name}"):
                try:
                    if uploaded_file.type.startswith("image"):
                        image_bytes = uploaded_file.read()
                        result_image = process_image(
                            image_bytes, predictor, model_thresholds, background_img,
                            replace_mode=replace_mode,
                            fill_color=fill_color,
                            blur_radius=blur_radius
                        )
                        st.image(result_image, caption=f"Обработанное {uploaded_file.name}")
                        buf = io.BytesIO()
                        result_image.save(buf, format="PNG")
                        st.download_button(
                            label="Скачать изображение",
                            data=buf.getvalue(),
                            file_name=f"processed_{uploaded_file.name}.png"
                        )
                    elif uploaded_file.type.startswith("video"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
                            temp.write(uploaded_file.read())
                            video_path = temp.name
                        processed_video_path = process_video(
                            video_path, predictor, model_thresholds, background_img,
                            replace_mode=replace_mode,
                            fill_color=fill_color,
                            blur_radius=blur_radius
                        )
                        st.video(processed_video_path)
                        with open(processed_video_path, "rb") as f:
                            st.download_button(
                                label="Скачать видео",
                                data=f.read(),
                                file_name=f"processed_{uploaded_file.name}"
                            )
                except Exception as e:
                    st.error(f"Ошибка при обработке {uploaded_file.name}: {e}")

if __name__ == "__main__":
    main()
