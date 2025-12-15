import os
import tempfile
import glob
import atexit
import platform
import subprocess
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
import torch
from torch.cuda import amp

# Внешние зависимости
try:
    from lama_cleaner.model_manager import ModelManager
    from lama_cleaner.schema import Config, HDStrategy, LDMSampler
    from transformers import AutoProcessor, AutoModelForCausalLM
    import onnxruntime  # Для YOLO
except Exception as e:
    ModelManager = None
    Config = None
    HDStrategy = None
    LDMSampler = None
    AutoProcessor = None
    AutoModelForCausalLM = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# Настройка логирования
logging.basicConfig(
    filename='watermark_remover.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------- Утилиты ----------
def safe_run(cmd_list, env=None):
    """Запустить команду безопасно."""
    try:
        result = subprocess.run(cmd_list, check=True, env=env, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f!Команда не выполнена: {e.stderr}")
        return None

@st.cache_resource(show_spinner=False)
def install_minimal_dependencies():
    """Установка системных пакетов."""
    system = platform.system()
    if system == "Linux":
        safe_run(["apt-get", "update"])
        safe_run(["apt-get", "install", "-y", "libgl1-mesa-glx", "ffmpeg"])
    return True

install_minimal_dependencies()

# ---------- Улучшенные модели ----------
class FlorenceModel:
    def __init__(self, model_id: str, precision: str = "float32"):
        if _IMPORT_ERROR:
            raise RuntimeError(f"Библиотеки не установлены: {_IMPORT_ERROR}")
        
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = precision
        
        # Загрузка с mixed precision
        dtype = torch.float16 if precision == "float16" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype
        ).to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

@st.cache_resource
def get_florence_model(model_id: str, precision: str):
    return FlorenceModel(model_id, precision)

class WatermarkRemover:
    def __init__(self, florence_model: FlorenceModel, use_gpu: bool = True):
        if _IMPORT_ERROR:
            raise RuntimeError(f"Библиотеки не установлены: {_IMPORT_ERROR}")
            
        self.model = florence_model
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Модель для уточнения маски
        self.yolo_model = None  # Загрузить при необходимости
        
        self.model_manager = ModelManager(
            name="lama",
            device=self.model.device,
            fp16=self.use_gpu
        )

    def _preprocess_mask(self, mask_cv: np.ndarray) -> np.ndarray:
        """Улучшенная обработка маски."""
        # Морфологические операции
        kernel = np.ones((3, 3), np.uint8)
        mask_cv = cv2.morphologyEx(mask_cv, cv2.MORPH_CLOSE, kernel)
        mask_cv = cv2.dilate(mask_cv, kernel, iterations=1)
        
        # Размытие границ
        mask_cv = cv2.GaussianBlur(mask_cv, (5, 5), 0)
        return mask_cv

    def process_image(self, image_cv: np.ndarray, mask_cv: np.ndarray, **kwargs) -> np.ndarray:
        """Обработка с mixed precision."""
        with amp.autocast(enabled=self.use_gpu):
            config = Config(
                ldm_steps=kwargs.get("steps", 20),
                ldm_sampler=kwargs.get("sampler", LDMSampler.ddim),
                hd_strategy=kwargs.get("strategy", HDStrategy.RESIZE),
                hd_strategy_crop_margin=kwargs.get("margin", 32),
                hd_strategy_crop_trigger_size=kwargs.get("trigger", 200),
                hd_strategy_resize_limit=kwargs.get("limit", 512),
            )
            result = self.model_manager(image_cv, mask_cv, config)
        return result

    def create_mask(self, image_pil: Image.Image, prediction: dict) -> Image.Image:
        """Создание маски с пост‑обработкой."""
        mask = Image.new("L", image_pil.size, 0)
        draw = ImageDraw.Draw(mask)
        
        for polygons in prediction.get("polygons", []):
            for poly in polygons:
                arr = np.array(poly).reshape(-1, 2)
                if len(arr) < 3:
                    continue
                coords = [tuple(map(float, xy)) for xy in arr]
                draw.polygon(coords, fill=255)
        
        # Размытие маски для плавного перехода
        mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
        return mask

    def run_florence_segmentation(self, image: Image.Image) -> dict:
        """Сегментация с улучшенной обработкой."""
        text_input = "watermark"
        task_prompt = "<REGION_TO_SEGMENTATION>"
        
        inputs = self.model.processor(
            text=task_prompt + text_input,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
                early_stopping=True
            )

        generated_text = self.model.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.model.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed.get("<REGION_TO_SEGMENTATION>", {})

    def process_image_advanced(self, image_path: str, output_path: str, **options) -> str:
        """Расширенная обработка с настройками."""
        try:
            image = Image.open(image_path).convert("RGB")
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Сегментация
            prediction = self.run_florence_segmentation(image)
            mask_pil = self.create_mask(image, prediction)
            mask_cv = np.array(mask_pil)

            if mask_cv.ndim == 2:
                mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_GRAY2BGR)

            # Улучшение маски
            mask_cv = self._preprocess_mask(mask_cv)

            # Обработка
            result_cv = self.process_image(
                image_cv,
                mask_cv,
                steps=options.get("steps", 20),
                sampler=options.get("sampler", LDMSampler.ddim),
                strategy=options.get("strategy", HDStrategy.RESIZE)
            )

                        result_pil = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
            
            # Сохранение
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            result_pil.save(output_path, quality=95, optimize=True)
            return output_path

        except Exception as e:
            logging.error(f"Ошибка обработки {image_path}: {e}")
            raise

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Watermark Remover Pro", layout="wide")
    st.title("Удаление водяных знаков — Pro-версия")

    if _IMPORT_ERROR:
        st.error(f"""
        Необходимые библиотеки не установлены: {_IMPORT_ERROR}
        
        **Как исправить:**
        1. Установите зависимости:  
           ```bash
           pip install lama-cleaner transformers onnxruntime-gpu
           ```
        2. Перезапустите приложение.
        """)
        return

    # Настройки обработки
    st.sidebar.header("Настройки")
    model_choices = [
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-base-ft",
        "microsoft/Florence-2-large",
        "microsoft/Florence-2-large-ft",
    ]
    selected_model = st.sidebar.selectbox(
        "Модель Florence", model_choices, index=2
    )
    
    precision = st.sidebar.radio(
        "Точность вычислений", ["float32", "float16"], index=1
        if torch.cuda.is_available() else 0
    )
    
    use_gpu = st.sidebar.checkbox("Использовать GPU", value=torch.cuda.is_available())
    
    # Расширенные параметры
    st.sidebar.subheader("Параметры удаления")
    steps = st.sidebar.slider("Шаги LDM", 1, 50, 20)
    strategy = st.sidebar.selectbox(
        "Стратегия HD", [HDStrategy.RESIZE, HDStrategy.CROP, HDStrategy.NONE]
    )
    
    with st.spinner("Загрузка модели..."):
        try:
            florence = get_florence_model(selected_model, precision)
            remover = WatermarkRemover(florence, use_gpu)
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            return

    # Основной интерфейс
    st.markdown("### 1. Обработка одного изображения")
    uploaded = st.file_uploader(
        "Загрузите изображение (PNG/JPG)", type=["png", "jpg", "jpeg"]
    )
    
    if uploaded:
        # Временный файл
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}")
        tmp_in.write(uploaded.read())
        tmp_in.flush()
        tmp_in.close()
        atexit.register(lambda: safe_remove(tmp_in.name))

        try:
            img = Image.open(tmp_in.name).convert("RGB")
            st.image(img, caption="Оригинал", use_column_width=True)
        except Exception as e:
            st.error(f"Ошибка открытия: {e}")
            safe_remove(tmp_in.name)
            return

        if st.button("Удалить водяной знак", key="single_process"):
            out_path = tmp_in.name.rsplit(".", 1)[0] + "_clean.png"
            with st.spinner("Обработка..."):
                try:
                    remover.process_image_advanced(
                        tmp_in.name,
                        out_path,
                        steps=steps,
                        strategy=strategy
                    )
                    st.success("Готово!")
                    
                    # Сравнение
                    col1, col2 = st.columns(2)
                    col1.image(img, caption="До")
                    col2.image(Image.open(out_path), caption="После")
                    
                    # Кнопка скачивания
                    with open(out_path, "rb") as file:
                        st.download_button(
                            label="Скачать результат",
                            data=file,
                            file_name=f!cleaned_{uploaded.name}",
                            mime="image/png"
                        )
                        
                except Exception as e:
                    st.error(f"Ошибка: {e}")
                    logging.exception(e)

    st.markdown("---")
    st.markdown("### 2. Пакетная обработка")
    
    col1, col2 = st.columns(2)
    with col1:
        input_folder = st.text_input("Входная папка", value="")
    with col2:
        output_folder = st.text_input("Папка для результатов", value="")

    max_workers = st.slider("Потоки", 1, 8, 4)
    
    if st.button("Запустить обработку", key="batch_process"):
        if not input_folder or not output_folder:
            st.error("Укажите обе папки!")
            return
            
        files = [
            p for p in glob.glob(os.path.join(input_folder, "*.*"))
            if p.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        
        if not files:
            st.info("Нет изображений для обработки.")
            return
            
        total = len(files)
        progress_bar = st.progress(0)
        status_text = st.empty()
        error_list = []
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Пакетная обработка
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    remover.process_image_advanced,
                    path,
                    os.path.join(
                        output_folder,
                        f"{Path(path).stem}_clean.png"
                    ),
                    steps=steps,
                    strategy=strategy
                ): path
                for path in files
            }
            
            completed = 0
            for future in as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    error_list.append(f"{path}: {str(e)}")
                
                completed += 1
                progress_bar.progress(completed / total)
                status_text.text(f"Обработано: {completed}/{total}")
        
        # Итоги
        if error_list:
            st.error(f"Ошибки ({len(error_list)}):")
            for err in error_list[:10]:
                st.code(err)
        else:
            st.success(f"Готово! Обработано {total} изображений.")

        # Логи
        st.markdown("#### Логи обработки")
        log_file = Path("watermark_remover.log")
        if log_file.exists():
            st.text(log_file.read_text()[-2000:])  # Последние 2000 символов

def safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logging.warning(f"Не удалось удалить {path}: {e}")

if __name__ == "__main__":
    main()
