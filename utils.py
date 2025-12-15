import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    """Преобразует изображение в массив NumPy."""
    return np.array(Image.open(image))

def show_result(result):
    """Показывает результат в Streamlit."""
    st.image(result, caption="Результат обработки", use_column_width=True)

def get_device():
    """Выбирает устройство для вычислений (GPU/CPU)."""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
