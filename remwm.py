import streamlit as st
from PIL import Image
import os
import tempfile
import torch
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import numpy as np
import subprocess
import atexit

# Установка пакета flash-attn
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

class WatermarkRemover:
    def __init__(self, model_id='microsoft/Florence-2-large'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.florence_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
        self.florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model_manager = ModelManager(name="lama", device=self.device)

    def process_image(self, image, mask, strategy=HDStrategy.RESIZE, sampler=LDMSampler.ddim, fx=1, fy=1):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if fx != 1 or fy != 1:
            image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
        config = Config(
            ldm_steps=1,
            ldm_sampler=sampler,
            hd_strategy=strategy,
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=200,
            hd_strategy_resize_limit=200,
        )
        result = self.model_manager(image, mask, config)
        return result

    def create_mask(self, image, prediction):
        mask = Image.new("RGBA", image.size, (0, 0, 0, 255))
        draw = ImageDraw.Draw(mask)
        scale = 1
        for polygons in prediction['polygons']:
            for _polygon in polygons:
                _polygon = np.array(_polygon).reshape(-1, 2)
                if len(_polygon) < 3:
                    continue
                _polygon = (_polygon * scale).reshape(-1).tolist()
                draw.polygon(_polygon, fill=(255, 255, 255, 255))
        return mask

    def process_images_florence_lama(self, input_image_path, output_image_path):
        image = Image.open(input_image_path).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        text_input = 'watermark'
        task_prompt = '<REGION_TO_SEGMENTATION>'
        inputs = self.florence_processor(text=task_prompt + text_input, images=image, return_tensors="pt").to(self.device)
        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        mask_image = self.create_mask(image, parsed_answer['<REGION_TO_SEGMENTATION>'])
        result_image = self.process_image(image_cv, np.array(mask_image), HDStrategy.RESIZE, LDMSampler.ddim)
        result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        result_image_pil.save(output_image_path)

# Создаем объект модели
model = WatermarkRemover()

st.title("Удаление водяных знаков с изображений")
uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])

# Обработка файла
if uploaded_file is not None:
    # Временные файлы для обработки
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_input:
        input_path = temp_input.name
        uploaded_file.seek(0)
        temp_input.write(uploaded_file.read())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_output:
        output_path = temp_output.name

    # Показываем загруженное изображение
    st.image(Image.open(input_path), caption='Загруженное изображение', use_column_width=True)

    # Кнопка для запуска обработки
    if st.button("Удалить водяной знак"):
        with st.spinner('Обработка изображения...'):
            try:
                model.process_images_florence_lama(input_path, output_path)
                result_image = Image.open(output_path)
                st.image(result_image, caption='Обработанное изображение', use_column_width=True)
            except Exception as e:
                st.error(f"Произошла ошибка: {e}")

# Очистка временных файлов
def cleanup_files():
    for filename in [input_path, output_path]:
        try:
            os.remove(filename)
        except Exception:
            pass

atexit.register(cleanup_files)
