import streamlit as st
import cv2
import numpy as np
from PIL import Image
from lightning_module import WatermarkDetection
from utils import preprocess_image, show_result

# Основная страница приложения
def main():
    st.title("Watermark Removal App")
    uploaded_file = st.file_uploader("Загрузите изображение:", type=['jpg', 'png'])
    
    if uploaded_file is not None:
        # Преобразовать изображение в массив NumPy
        image_array = preprocess_image(uploaded_file)
        
        # Применение модели
        model = WatermarkDetection.load_from_checkpoint(checkpoint_path='./checkpoints/best_model.ckpt')
        cleaned_image = model.remove_watermark(image_array)
        
        # Показываем результат
        show_result(cleaned_image)

if __name__ == "__main__":
    main()
