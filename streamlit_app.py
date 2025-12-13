import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
from moviepy.editor import VideoFileClip, ImageSequenceClip
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch

@st.cache(allow_output_mutation=True)
def load_mask_rcnn_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    return predictor

def detect_masks(image_cv2, predictor, threshold):
    outputs = predictor(image_cv2)
    masks = outputs["instances"].pred_masks.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    selected_masks = [m for m, s in zip(masks, scores) if s >= threshold]
    return selected_masks

def remove_watermarks(image, masks, background_img=None):
    image_np = np.array(image)
    if background_img is not None:
        background = np.array(background_img.resize(image.size))
    else:
        mean_color = np.mean(image_np, axis=(0, 1)).astype(np.uint8)
        background = np.full_like(image_np, mean_color)
    for mask in masks:
        mask_bool = mask.astype(bool)
        image_np[mask_bool] = background[mask_bool]
    return Image.fromarray(image_np)

def process_image(image_bytes, predictor, threshold, background_img=None):
    image_cv2 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    masks = detect_masks(image_cv2, predictor, threshold)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if masks:
        image = remove_watermarks(image, masks, background_img)
    return image

def process_video(video_path, predictor, threshold, background_img=None):
    clip = VideoFileClip(video_path)
    processed_frames = []

    for frame in clip.iter_frames():
        frame_bytes = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes()
        processed_frame = process_image(frame_bytes, predictor, threshold, background_img)
        processed_frames.append(np.array(processed_frame))
    processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    processed_clip.write_videofile(output_path, codec="libx264", logger=None)
    return output_path

def main():
    # Вставка кастомных стилей для улучшения дизайна
    st.markdown(
        """
        <style>
        /* Общий фон */
        body {
            background-color: #f0f2f6;
        }
        /* Заголовки */
        h1 {
            color: #4A90E2;
        }
        /* Разделы */
        .section {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        /* Кнопки */
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            font-weight: bold;
        }
        /* Подсказки */
        .stMarkdown {
            font-family: 'Arial', sans-serif;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("🖼️✨ Улучшенное удаление водяных знаков + настраиваемый фон")
    st.markdown("## 🚀 Быстрый старт")
    st.write("Эта программа автоматически обнаружит и удалит водяные знаки на изображениях и видео. "
             "Вы можете загрузить свой фон, чтобы вставить его вместо водяных знаков. Используйте настройки для оптимизации результатов.")

    # Создаем разделы
    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("🔌 Загрузка модели")
        if st.button("Загрузить модель для обнаружения", key="load_model"):
            with st.spinner("Идет загрузка модели..."):
                predictor = load_mask_rcnn_model()
            st.success("Модель успешно загружена!")
        else:
            predictor = None
        st.markdown('</div>', unsafe_allow_html=True)

    if predictor is None:
        st.info("Пожалуйста, нажмите кнопку 'Загрузить модель', чтобы продолжить.")
        return

    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("🎚️ Настройки обработки")
        threshold = st.slider("Порог обнаружения маски", 0.0, 1.0, 0.5, 0.05,
                              help="Чем ниже значение, тем больше объектов будет удалено.")
        mode = st.selectbox("Режим обработки", ["Баланс", "Быстрая", "Качество"],
                            help="Выберите режим для настройки порога и скорости.")
        if mode == "Быстрая":
            threshold = 0.3
        elif mode == "Качество":
            threshold = 0.7
        st.write(f"Текущий порог: {threshold:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("🖼️ Выбор фона")
        background_file = st.file_uploader("Загрузите изображение фона", type=["png", "jpg", "jpeg"])
        background_img = None
        if background_file:
            background_img = Image.open(background_file).convert("RGB")
            st.image(background_img, caption="Выбранный фон", width=300)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("📂 Загрузка файлов для обработки")
        uploaded_files = st.file_uploader(
            "Выберите изображения или видео для обработки",
            type=["png", "jpg", "jpeg", "mp4", "avi"],
            accept_multiple_files=True
        )
        if not uploaded_files:
            st.warning("Пожалуйста, загрузите файлы для начала обработки.")
        else:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Обработка файла: {uploaded_file.name}"):
                    try:
                        if uploaded_file.type.startswith("image"):
                            image_bytes = uploaded_file.read()
                            result_image = process_image(image_bytes, predictor, threshold, background_img)
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
                            processed_video_path = process_video(video_path, predictor, threshold, background_img)
                            st.video(processed_video_path)
                            with open(processed_video_path, "rb") as f:
                                st.download_button(
                                    label="Скачать видео",
                                    data=f.read(),
                                    file_name=f"processed_{uploaded_file.name}"
                                )
                    except Exception as e:
                        st.error(f"Ошибка при обработке {uploaded_file.name}: {e}")

    st.success("Обработка завершена! Спасибо за использование! 😊")

if __name__ == "__main__":
    main()
