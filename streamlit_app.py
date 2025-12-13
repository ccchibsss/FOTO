import sys
import subprocess
import os
import io
import pickle
import concurrent.futures
import tempfile

def install_package(package_name):
    """Устанавливает пакет через pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])

def check_and_install_dependencies():
    """Проверяет и устанавливает необходимые зависимости без distutils"""
    # Проверка и установка setuptools и wheel
    try:
        import setuptools
        import wheel
    except ModuleNotFoundError:
        print("Установка setuptools и wheel...")
        install_package("setuptools")
        install_package("wheel")
    else:
        print("Обновление setuptools и wheel...")
        install_package("setuptools")
        install_package("wheel")
    
    # Обновление pip
    print("Обновление pip...")
    install_package("pip")
    
    # Попытка установить torch
    try:
        import torch
        print("PyTorch уже установлен.")
    except ImportError:
        print("PyTorch не установлен. Попытка установки...")
        try:
            install_package("torch")
            import torch
            print("PyTorch успешно установлен.")
        except Exception as e:
            print(f"Ошибка установки torch: {e}")
            print("Возможно, для вашей версии Python wheel для torch отсутствует.")
            print("Рекомендуется использовать Python 3.10 или 3.11 или установить torch вручную из https://pytorch.org/")

if __name__ == "__main__":
    check_and_install_dependencies()

# Далее ваш оригинальный код
import torch
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFilter
from moviepy.editor import VideoFileClip, ImageSequenceClip

# --- Вариант 1: detectron2 ---
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    detectron2_available = True
except ImportError:
    detectron2_available = False

# --- Вариант 2: собственная модель ---
from torchvision.models.detection import MaskRCNNPredictor

# --- Класс для кастомных моделей ---
class CustomModel:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_cv2):
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            outputs = self.model([tensor.to(self.device)])
        masks = (outputs[0]['masks'] > 0.5).squeeze(1).cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        return list(masks), list(scores)

# --- Основной класс обработки ---
class WatermarkProcessor:
    def __init__(self):
        # Конфигурация моделей
        self.models_config = {
            "Mask R-CNN ResNet50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            "Mask R-CNN ResNet101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            "Custom Model": None  # путь к вашей модели
        }
        self.predictors = {}
        self.load_models()
        self.settings = {
            "model_name": "Mask R-CNN ResNet50",
            "threshold": 0.5,
            "replace_mode": "auto",
            "fill_color": "#000000",
            "blur_radius": 15,
            "custom_model_path": None
        }

    def load_models(self):
        if detectron2_available:
            for name, config_path in self.models_config.items():
                if config_path:
                    self.predictors[name] = self.load_detectron2_model(config_path)
        # Загрузка кастомной модели
        self.custom_model = None
        if self.settings["model_name"] == "Custom Model" and self.settings["custom_model_path"]:
            self.custom_model = CustomModel(self.settings["custom_model_path"], device='cuda' if torch.cuda.is_available() else 'cpu')

    def load_detectron2_model(self, config_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.settings["threshold"]
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.DEVICE = device
        predictor = DefaultPredictor(cfg)
        return predictor

    def detect_masks(self, image_cv2, model_name, threshold):
        if model_name == "Custom Model" and self.custom_model:
            masks, scores = self.custom_model.predict(image_cv2)
            return masks, scores
        elif model_name in self.predictors:
            predictor = self.predictors[model_name]
            outputs = predictor(image_cv2)
            masks = outputs["instances"].pred_masks.cpu().numpy()
            scores = outputs["instances"].scores.cpu().numpy()
            selected_masks = [mask for mask, score in zip(masks, scores) if score >= threshold]
            selected_scores = [score for score in scores if score >= threshold]
            return selected_masks, selected_scores
        else:
            return [], []

    def auto_decision(self, masks, scores, shape):
        if not masks:
            return "Цвет"
        total_area = sum(np.sum(mask) for mask in masks)
        mask_density = total_area / (shape[0] * shape[1])
        max_mask_size = max(np.sum(mask) for mask in masks)
        max_mask_norm = max_mask_size / (shape[0] * shape[1])
        avg_score = np.mean(scores)

        if mask_density > 0.2:
            return "Размытие"
        elif max_mask_norm < 0.01 and avg_score < 0.5:
            return "Цвет"
        elif mask_density > 0.1 and max_mask_norm > 0.05:
            return "Размытие"
        else:
            return "Цвет"

    def remove_watermarks(self, image, masks, mode="Цвет", color="#000000", blur_radius=5):
        image_np = np.array(image)
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

    def process_image(self, image_bytes, model_name, background_img, replace_mode, fill_color, blur_radius):
        image_cv2 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        threshold = self.settings["threshold"]
        masks, scores = self.detect_masks(image_cv2, model_name, threshold)
        shape = image_cv2.shape[:2]
        auto_mode = self.auto_decision(masks, scores, shape)
        mode = auto_mode if replace_mode == "auto" else replace_mode

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Предварительный просмотр маски
        if masks:
            mask_overlay = np.zeros_like(np.array(image))
            for mask in masks:
                mask_overlay[mask.astype(bool)] = [255, 0, 0]
            overlay_img = Image.fromarray(mask_overlay).convert("RGBA")
            preview = image.convert("RGBA")
            combined = Image.alpha_composite(preview, overlay_img)
            st.image(combined, caption="Маска (красный — водяной знак)")
            if st.button("Удалить водяной знак с этим изображением?"):
                image = self.remove_watermarks(image, masks, mode=mode, color=fill_color, blur_radius=blur_radius)
        else:
            st.info("Объекты не обнаружены.")
        return image

    def process_video(self, video_path, background_img, replace_mode, fill_color, blur_radius):
        clip = VideoFileClip(video_path)
        frames = list(clip.iter_frames())

        processed_frames = []

        def worker(frame):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bytes = cv2.imencode('.jpg', frame_bgr)[1].tobytes()
            processed_img = self.process_image(frame_bytes, self.settings["model_name"], background_img, replace_mode, fill_color, blur_radius)
            return np.array(processed_img)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, f) for f in frames]
            for future in concurrent.futures.as_completed(futures):
                processed_frames.append(future.result())

        processed_frames = [Image.fromarray(frame) for frame in processed_frames]
        processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        processed_clip.write_videofile(output_path, codec="libx264", logger=None)
        return output_path

    def save_settings(self, filename='settings.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.settings, f)

    def load_settings(self, filename='settings.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.settings = pickle.load(f)

# --- Streamlit UI ---
def main():
    st.title("🖼️✨ Массовая обработка изображений и видео — водяные знаки")
    processor = WatermarkProcessor()

    # Выбор модели
    model_name = st.selectbox("Модель обнаружения водяных знаков", list(processor.models_config.keys()))
    processor.settings["model_name"] = model_name

    # Загрузка кастомной модели
    if model_name == "Custom Model":
        uploaded_model = st.file_uploader("Загрузите вашу модель (.pth)", type=["pth"])
        if uploaded_model:
            os.makedirs("models", exist_ok=True)
            temp_model_path = os.path.join("models", uploaded_model.name)
            with open(temp_model_path, "wb") as f:
                f.write(uploaded_model.read())
            processor.settings["custom_model_path"] = temp_model_path
            # Перезагружаем модель
            processor.load_models()

    # Настройки порогов
    st.markdown("## 🔧 Настройки порогов")
    for m in processor.models_config:
        default = processor.settings.get("threshold", 0.5)
        threshold = st.slider(f"Порог для {m}", 0.0, 1.0, default, 0.05)
        processor.settings[f"threshold_{m}"] = threshold
    # Обновляем основной threshold
    processor.settings["threshold"] = processor.settings.get(f"threshold_{model_name}", 0.5)

    # Остальные параметры
    replace_mode = st.selectbox("Режим обработки", ["auto", "Цвет", "Размытие"])
    fill_color = st.color_picker("Цвет заливки", "#000000")
    blur_radius = st.slider("Радиус размытия", 1, 25, 15)

    # Загрузка фона
    background_file = st.file_uploader("Фон (опционально)", type=["png", "jpg", "jpeg"])
    background_img = None
    if background_file:
        background_img = Image.open(background_file).convert("RGB")
        st.image(background_img, caption="Выбранный фон", width=200)

    # Загрузка файлов
    uploaded_files = st.file_uploader("Выберите файлы (изображения или видео)", type=["png", "jpg", "jpeg", "mp4"], accept_multiple_files=True)
    if uploaded_files:
        for upf in uploaded_files:
            with st.spinner(f"Обработка {upf.name}"):
                try:
                    if upf.type.startswith("image"):
                        image_bytes = upf.read()
                        result_img = processor.process_image(
                            image_bytes,
                            model_name,
                            background_img,
                            replace_mode,
                            fill_color,
                            blur_radius
                        )
                        st.image(result_img)
                        buf = io.BytesIO()
                        result_img.save(buf, format="PNG")
                        st.download_button("Скачать изображение", buf.getvalue(), file_name=f"processed_{upf.name}.png")
                    elif upf.type.startswith("video"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                            tmp_in.write(upf.read())
                            video_path = tmp_in.name
                        output_path = processor.process_video(video_path, background_img, replace_mode, fill_color, blur_radius)
                        st.video(output_path)
                        with open(output_path, "rb") as f:
                            st.download_button("Скачать видео", f.read(), file_name=f"processed_{upf.name}")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    # Сохранение/загрузка настроек
    if st.button("Сохранить настройки"):
        processor.save_settings()
        st.success("Настройки сохранены")
    if st.button("Загрузить настройки"):
        processor.load_settings()
        st.success("Настройки загружены")
        st.json(processor.settings)

if __name__ == "__main__":
    main()
