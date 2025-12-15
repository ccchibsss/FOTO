import streamlit as st
import os
import tempfile
import threading
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import subprocess
import platform

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler
from transformers import AutoProcessor, AutoModelForCausalLM

# --- Установка зависимостей и системных библиотек ---
@st.cache(allow_output_mutation=True)
def install_dependencies():
    system = platform.system()
    if system == "Linux":
        # Для Linux
        try:
            subprocess.run('apt-get update', shell=True, check=True)
            subprocess.run('apt-get install -y libgl1-mesa-glx', shell=True, check=True)
        except Exception:
            # Можно логировать ошибку или показывать сообщение
            pass
    elif system == "Windows":
        # Для Windows
        st.info(
            "На Windows рекомендуется обновить драйвер видеокарты и установить последние версии DirectX "
            "для корректной работы. Обновите драйверы видеокарты через сайт производителя."
        )
    else:
        # Для других систем (macOS и т.п.) — ничего не делаем
        pass

    # Установка Python-библиотек
    try:
        subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True, check=True)
    except subprocess.CalledProcessError:
        # Можно логировать ошибку или показывать сообщение
        pass

# Вызываем установку один раз при запуске
install_dependencies()

# --- Классы моделей ---
class FlorenceModel:
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

class WatermarkRemover:
    def __init__(self, model: FlorenceModel):
        self.model = model
        self.model_manager = ModelManager(name="lama", device=self.model.device)

    def process_image(self, image, mask, strategy=HDStrategy.RESIZE, sampler=LDMSampler.ddim, fx=1, fy=1):
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask_cv = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if fx != 1 or fy != 1:
            image_cv = cv2.resize(image_cv, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
            mask_cv = cv2.resize(mask_cv, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
        config = Config(
            ldm_steps=1,
            ldm_sampler=sampler,
            hd_strategy=strategy,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=200,
            hd_strategy_resize_limit=200,
        )
        result = self.model_manager(image_cv, mask_cv, config)
        return result

    def create_mask(self, image, prediction):
        mask = Image.new("RGBA", image.size, (0, 0, 0, 255))
        draw = ImageDraw.Draw(mask)
        scale = 1
        for polygons in prediction.get('polygons', []):
            for _polygon in polygons:
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    continue
                _polygon = (_polygon * scale).reshape(-1).tolist()
                draw.polygon(_polygon, fill=(255, 255, 255, 255))
        return mask

    def run_florence_segmentation(self, image):
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        text_input = 'watermark'
        task_prompt = '<REGION_TO_SEGMENTATION>'
        inputs = self.model.processor(text=task_prompt + text_input, images=image_pil, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.model.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.model.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image_pil.width, image_pil.height)
        )
        return parsed_answer.get('<REGION_TO_SEGMENTATION>', {})

    def process_florence_image(self, image_path, output_path):
        image = Image.open(image_path).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        prediction = self.run_florence_segmentation(image)
        mask_image = self.create_mask(image, prediction)
        result = self.process_image(image_cv, np.array(mask_image), HDStrategy.RESIZE, LDMSampler.ddim)
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        result_pil.save(output_path)
        return output_path

# --- Основная логика ---
def main():
    # Модельные опции
    model_choices = [
        'microsoft/Florence-2-base',
        'microsoft/Florence-2-base-ft',
        'microsoft/Florence-2-large',
        'microsoft/Florence-2-large-ft'
    ]

    # Кеш моделей
    models_cache = {}
    for m_id in model_choices:
        models_cache[m_id] = FlorenceModel(m_id)

    def get_remover(model_id):
        return WatermarkRemover(models_cache[model_id])

    st.title("Удаление водяных знаков с изображений")
    st.write("Выберите модель Florence для сегментации и загрузите изображение.")

    # Выбор модели
    selected_model_id = st.selectbox("Модель Florence", options=model_choices, index=2)

    # Загрузка файла
    uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_input:
            input_path = temp_input.name
            uploaded_file.seek(0)
            temp_input.write(uploaded_file.read())

        remover = get_remover(selected_model_id)

        # Отображение загруженного изображения
        original_img = Image.open(input_path)
        st.image(original_img, caption='Загруженное изображение', use_column_width=True)

        # Обработка по кнопке
        if st.button("Удалить водяной знак"):
            with st.spinner('Обработка изображения...'):
                try:
                    output_path = input_path.replace('.jpg', '_result.png').replace('.png', '_result.png')
                    remover.process_florence_image(input_path, output_path)
                    result_img = Image.open(output_path)
                    st.image(result_img, caption='Обработанное изображение', use_column_width=True)
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        # Очистка временных файлов
        def cleanup():
            try:
                os.remove(input_path)
                os.remove(output_path)
            except:
                pass
        st.on_event("close", cleanup)

    # --- Пакетная обработка ---
    st.write("---")
    st.subheader("Пакетная обработка папки")
    folder_path = st.text_input("Путь к папке с изображениями")
    output_folder = st.text_input("Путь к папке для сохранения")
    max_workers = st.number_input("Число потоков", min_value=1, max_value=8, value=4)

    if st.button("Обработать папку"):
        if folder_path and output_folder:
            files = glob.glob(os.path.join(folder_path, "*.*"))
            total_files = len(files)
            st.write(f"Обработка {total_files} изображений...")

            def process_batch():
                for idx, file_path in enumerate(files, 1):
                    filename = os.path.basename(file_path)
                    out_path = os.path.join(output_folder, filename)
                    try:
                        remover = get_remover(selected_model_id)
                        remover.process_florence_image(file_path, out_path)
                    except Exception as e:
                        st.write(f"Ошибка при обработке {filename}: {e}")
                    st.progress(idx / total_files)
                st.success("Обработка завершена!")

            threading.Thread(target=process_batch).start()
        else:
            st.error("Пожалуйста, укажите пути к папкам.")

if __name__ == "__main__":
    main()
