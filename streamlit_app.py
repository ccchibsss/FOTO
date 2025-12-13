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
    return [m for m, s in zip(masks, scores) if s >= threshold]

def remove_watermarks(image, masks):
    image_np = np.array(image)
    mean_color = np.mean(image_np, axis=(0, 1)).astype(np.uint8)
    for mask in masks:
        mask_bool = mask.astype(bool)
        # Можно добавить сглаживание или смешивание
        image_np[mask_bool] = mean_color
    return Image.fromarray(image_np)

def process_image(image_bytes, predictor, threshold):
    image_cv2 = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    masks = detect_masks(image_cv2, predictor, threshold)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if masks:
        image = remove_watermarks(image, masks)
    return image

def process_video(video_path, predictor, threshold):
    clip = VideoFileClip(video_path)
    processed_frames = []

    for frame in clip.iter_frames():
        frame_bytes = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1].tobytes()
        processed_frame = process_image(frame_bytes, predictor, threshold)
        processed_frames.append(np.array(processed_frame))
    processed_clip = ImageSequenceClip(processed_frames, fps=clip.fps)
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    processed_clip.write_videofile(output_path, codec="libx264", logger=None)
    return output_path

def main():
    st.title("Автоматическое удаление водяных знаков")
    st.sidebar.header("Настройки")
    threshold = st.sidebar.slider("Порог обнаружения маски", 0.0, 1.0, 0.5, 0.05)

    predictor = load_mask_rcnn_model()

    uploaded_files = st.file_uploader("Выберите файлы для обработки", type=["png", "jpg", "jpeg", "mp4", "avi"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Обработка файла: {uploaded_file.name}"):
                    if uploaded_file.type.startswith("image"):
                        image_bytes = uploaded_file.read()
                        result_image = process_image(image_bytes, predictor, threshold)
                        st.image(result_image, caption=f"Обработанное {uploaded_file.name}")
                        buf = io.BytesIO()
                        result_image.save(buf, format="PNG")
                        st.download_button(f"Скачать {uploaded_file.name}", data=buf.getvalue(), file_name=f"processed_{uploaded_file.name}.png")
                    elif uploaded_file.type.startswith("video"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
                            temp.write(uploaded_file.read())
                            video_path = temp.name
                        processed_video_path = process_video(video_path, predictor, threshold)
                        st.video(processed_video_path)
                        with open(processed_video_path, "rb") as f:
                            st.download_button(f"Скачать {uploaded_file.name}", data=f.read(), file_name=f"processed_{uploaded_file.name}")
            except Exception as e:
                st.error(f"Ошибка при обработке {uploaded_file.name}: {e}")
    else:
        st.info("Пожалуйста, загрузите файлы для обработки.")

if __name__ == "__main__":
    main()
