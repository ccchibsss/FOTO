import streamlit as st
import os
import torch
import numpy as np
from PIL import Image
import cv2
import requests
from pathlib import Path

# --- Настройки ---
st.set_page_config(page_title="Удаление водяных знаков", layout="wide")
st.title("🧼 Удаление водяных знаков с изображений")

# Папка для моделей
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# URLs моделей
U2NET_URL = "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0.0/u2net.pth"
LAMA_URL = "https://huggingface.co/aimetis/lama/resolve/main/big-lama/models/best.ckpt"

U2NET_PATH = models_dir / "u2net.pth"
LAMA_PATH = models_dir / "lama.pth"  # переименовываем .ckpt → .pth

# --- Функция: скачать модель ---
def download_model(url: str, path: Path, name: str):
    if path.exists():
        st.info(f"✅ {name} уже загружена.")
        return True

    st.info(f"📥 Загружаю {name}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(path, 'wb') as f, st.spinner(f"Скачивание {name}..."):
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        st.progress(progress)
        st.success(f"✅ {name} успешно загружена.")
        return True
    except Exception as e:
        st.error(f"❌ Ошибка загрузки {name}: {e}")
        return False

# --- Скачиваем модели ---
if not download_model(U2NET_URL, U2NET_PATH, "U2-Net"):
    st.stop()

if not download_model(LAMA_URL, LAMA_PATH, "LaMa (big-lama)"):
    st.stop()

# --- Загрузка моделей ---
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"🔧 Используем устройство: {device}")

    # U2-Net
    try:
        import u2net
        model_u2net = u2net.U2NET(3, 1)
        model_u2net.load_state_dict(torch.load(U2NET_PATH, map_location=device))
        model_u2net.to(device).eval()
    except Exception as e:
        st.error(f"❌ Ошибка загрузки U2-Net: {e}")
        return None, device

    # LaMa
    try:
        from lama_cleaner.model_manager import ModelManager
        from lama_cleaner.schema import Config
        model_lama = ModelManager(name="lama", device=device)
        config = Config(indoor=False)
    except Exception as e:
        st.error(f"❌ Ошибка инициализации LaMa: {e}")
        return None, device

    return (model_u2net, model_lama, config), device

# --- Загружаем модели ---
models_data, device = load_models()
if models_data is None:
    st.stop()

u2net_model, lama_model, lama_config = models_data

st.success("✅ Все модели загружены и готовы к работе!")

# --- u2net.py (внутри кода!) ---
# Подключаем реализацию U2NET, если файла нет
U2NET_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

def _upsample_like(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = REBNCONV(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = REBNCONV(64, 64)
        self.outconv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        hx = x
        hx1 = self.stage1(hx)
        hx = self.pool1(hx1)
        hx2 = self.stage2(hx)
        d1 = self.outconv(hx1)
        return torch.sigmoid(d1),
'''

# Проверим, есть ли u2net.py
if not os.path.exists("u2net.py"):
    with open("u2net.py", "w") as f:
        f.write(U2NET_CODE)
    st.info("📝 Файл `u2net.py` создан автоматически.")

# Перезагружаем модуль
import importlib
import u2net
importlib.reload(u2net)

# --- Функция сегментации ---
def segment_watermark(image: Image.Image):
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred = u2net_model(img_tensor)[0]
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()
    
    mask = (pred * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

# --- Интерфейс загрузки ---
uploaded_file = st.file_uploader("📷 Загрузите изображение", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Оригинал")
        st.image(image, use_column_width=True)

    if st.button("🧹 Удалить водяной знак"):
        with st.spinner("Обработка..."):

            # Получаем маску
            mask = segment_watermark(image)

            # Восстановление
            try:
                input_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                result_bgr = lama_model(image=input_bgr, mask=mask, config=lama_config)
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                result_image = Image.fromarray(result_rgb)

                with col2:
                    st.subheader("Результат")
                    st.image(result_image, use_column_width=True)

                # Подготовка к скачиванию
                from io import BytesIO
                buf = BytesIO()
                result_image.save(buf, format="PNG")
                byte_img = buf.getvalue()

                st.download_button(
                    label="⬇️ Скачать очищенное изображение",
                    data=byte_img,
                    file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png"
                )

                with st.expander("🔍 Показать маску"):
                    st.image(mask, width=300, caption="Маска водяного знака")

            except Exception as e:
                st.error(f"❌ Ошибка при восстановлении: {e}")
