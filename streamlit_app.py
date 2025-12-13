import torch
import streamlit as st
import cv2
import numpy as np
import io
import os
import tempfile
import concurrent.futures
from PIL import Image, ImageFilter
from moviepy.editor import VideoFileClip, ImageSequenceClip
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T

# Класс для сегментации водяных знаков с Mask R-CNN
class WatermarkMaskRCNN:
    def __init__(self, threshold=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = maskrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])
        self.threshold = threshold

    def detect_masks(self, image_cv2):
        # Конвертация из BGR в RGB
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb).to(self.device)
        with torch.no_grad():
            outputs = self.model([tensor])
        masks = []
        scores = []
        for mask, score in zip(outputs[0]['masks'], outputs[0]['scores']):
            if score >= self.threshold:
                mask_np = mask.squeeze().cpu().numpy()
                binary_mask = mask_np > 0.5
                masks.append(binary_mask)
                scores.append(score.cpu().item())
        return masks, scores

    def remove_watermarks(self, image, masks, mode="auto", color="#000000", blur_radius=15):
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

# Основной класс обработки
class WatermarkProcessor:
    def __init__(self):
        self.threshold = 0.5
        self.model = WatermarkMaskRCNN(threshold=self.threshold)

    def detect_masks(self, image_cv2):
        return self.model.detect_masks(image_cv2)

    def remove_watermarks(self, image, masks, mode="auto", color="#000000", blur_radius=15):
        return self.model.remove_watermarks(image, masks, mode, color, blur_radius)

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

    def process_image(self, image_bytes, replace_mode, fill_color, blur_radius):
        image_cv2 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        masks, scores = self.detect_masks(image_cv2)
        shape = image_cv2.shape[:2]
        mode_decision = self.auto_decision(masks, scores, shape)
        mode = replace_mode if replace_mode != "auto" else mode_decision

        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Обязательно показываем маски
        if masks:
            mask_overlay = np.zeros_like(np.array(image_pil))
            for mask in masks:
                mask_overlay[mask.astype(bool)] = [255, 0, 0]
            overlay_img = Image.fromarray(mask_overlay).convert("RGBA")
            preview = image_pil.convert("RGBA")
            combined = Image.alpha_composite(preview, overlay_img)
            st.image(combined, caption="Маска (красный — водяной знак)")
            if st.button("Удалить водяной знак с этим изображением?"):
                image_pil = self.remove_watermarks(image_pil, masks, mode=mode, color=fill_color, blur_radius=blur_radius)
        else:
            st.info("Объекты не обнаружены.")
        return image_pil

    def process_video(self, video_path, replace_mode, fill_color, blur_radius):
        clip = VideoFileClip(video_path)
        frames = list(clip.iter_frames())

        def worker(frame):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bytes = cv2.imencode('.jpg', frame_bgr)[1].tobytes()
            processed_img = self.process_image(frame_bytes, replace_mode, fill_color, blur_radius)
            return np.array(processed_img)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, f) for f in frames]
            processed_frames = [f.result() for f in futures]

        processed_frames = [Image.fromarray(frame) for frame in processed_frames]
        output_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        output_clip.write_videofile(output_path, codec="libx264", logger=None)
        return output_path

# --- Ваш основной код Streamlit --- 
import pickle
import io
import os

def main():
    st.title("🖼️✨ Массовая обработка изображений и видео — водяные знаки")
    processor = WatermarkProcessor()

    # Выбор режима
    replace_mode = st.selectbox("Режим обработки", ["auto", "Цвет", "Размытие"])
    fill_color = st.color_picker("Цвет заливки", "#000000")
    blur_radius = st.slider("Радиус размытия", 1, 25, 15)

    # Загрузка файла
    uploaded_files = st.file_uploader("Выберите файлы (изображения или видео)", type=["png", "jpg", "jpeg", "mp4"], accept_multiple_files=True)
    if uploaded_files:
        for upf in uploaded_files:
            with st.spinner(f"Обработка {upf.name}"):
                try:
                    if upf.type.startswith("image"):
                        image_bytes = upf.read()
                        result_img = processor.process_image(image_bytes, replace_mode, fill_color, blur_radius)
                        st.image(result_img)
                        buf = io.BytesIO()
                        result_img.save(buf, format="PNG")
                        st.download_button("Скачать изображение", buf.getvalue(), file_name=f"processed_{upf.name}.png")
                    elif upf.type.startswith("video"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                            tmp_in.write(upf.read())
                            video_path = tmp_in.name
                        output_path = processor.process_video(video_path, replace_mode, fill_color, blur_radius)
                        st.video(output_path)
                        with open(output_path, "rb") as f:
                            st.download_button("Скачать видео", f.read(), file_name=f"processed_{upf.name}")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    # Настройки
    if st.button("Сохранить настройки"):
        with open("settings.pkl", "wb") as f:
            pickle.dump({
                "replace_mode": replace_mode,
                "fill_color": fill_color,
                "blur_radius": blur_radius
            }, f)
        st.success("Настройки сохранены")
    if st.button("Загрузить настройки"):
        if os.path.exists("settings.pkl"):
            with open("settings.pkl", "rb") as f:
                settings = pickle.load(f)
            replace_mode = settings.get("replace_mode", replace_mode)
            fill_color = settings.get("fill_color", fill_color)
            blur_radius = settings.get("blur_radius", blur_radius)
            st.success("Настройки загружены")

if __name__ == "__main__":
    main()
