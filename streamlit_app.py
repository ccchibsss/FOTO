# fixed_watermark_app.py
# Обновлённая и улучшенная версия вашего приложения для удаления водяных знаков.

import os
import sys
import io
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
import numpy as np
import cv2

# Попытка импортировать streamlit
try:
    import streamlit as st  # type: ignore
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

# Попытка импортировать Lightning модель
try:
    from lightning_module import WatermarkDetection
    LIGHTNING_AVAILABLE = True
except Exception:
    WatermarkDetection = None
    LIGHTNING_AVAILABLE = False

logging.basicConfig(level=logging.INFO, filename="watermark_app.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------- Утилиты -----------------
def open_image(file_or_path) -> Image.Image:
    """Открывает изображение PIL из файла или байтов."""
    if isinstance(file_or_path, str):
        return Image.open(file_or_path).convert("RGB")
    try:
        file_or_path.seek(0)
    except Exception:
        pass
    return Image.open(file_or_path).convert("RGB")


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    bio = io.BytesIO()
    img.save(bio, format=fmt)
    return bio.getvalue()


def detect_watermark_mask_cv(image_cv: np.ndarray) -> np.ndarray:
    """Обнаружение маски водяного знака с улучшениями."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Используем адаптивную пороговую обработку для выделения водяных знаков
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    # Размытие для сглаживания
    blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Удаление мелких компонентов
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = (image_cv.shape[0] * image_cv.shape[1]) * 0.0003
    out = np.zeros_like(mask)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def inpaint_image_cv(image_cv: np.ndarray, mask_cv: np.ndarray, method: str = "telea") -> np.ndarray:
    """Inpainting с параметрами и сглаживанием."""
    if mask_cv.ndim == 3:
        mask = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY)
    else:
        mask = mask_cv
    mask_bin = (mask > 0).astype("uint8") * 255
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    radius = 5  # Можно увеличить радиус для более "гладкого" результата
    try:
        inpainted = cv2.inpaint(image_cv, mask_bin, radius, flag)
    except Exception:
        # В случае ошибки — простое размытие
        inpainted = cv2.GaussianBlur(image_cv, (21, 21), 0)
        inpainted[mask_bin == 255] = inpainted[mask_bin == 255]
    # Применяем сглаживание
    smoothed = cv2.GaussianBlur(inpainted, (5, 5), 0)
    return smoothed


def smoothen_image(image: np.ndarray) -> np.ndarray:
    """Дополнительное сглаживание."""
    return cv2.GaussianBlur(image, (5, 5), 0)


# ----------------- подготовка изображений -----------------
def preprocess_image(file_or_path) -> np.ndarray:
    """Возвращает изображение как numpy RGB-массив."""
    img = open_image(file_or_path)
    arr = np.array(img)
    return arr


def show_result(original: np.ndarray, cleaned: np.ndarray, title_original: str = "Original",
                title_cleaned: str = "Cleaned"):
    """Показ результата в Streamlit или сохранение файла в CLI."""
    orig_pil = Image.fromarray(original)
    clean_pil = Image.fromarray(cleaned)
    if ST_AVAILABLE:
        col1, col2 = st.columns(2)
        col1.image(orig_pil, caption=title_original, use_column_width=True)
        col2.image(clean_pil, caption=title_cleaned, use_column_width=True)
        return pil_to_bytes(clean_pil, fmt="PNG")
    else:
        out_path = Path(tempfile_filename("cleaned_"))
        clean_pil.save(out_path, format="PNG")
        print("Сохранено изображение:", out_path)
        return None


def tempfile_filename(prefix: str = "tmp_", suffix: str = ".png") -> str:
    import tempfile
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path


# ----------------- модель и fallback -----------------
class WatermarkDetectionFallback:
    """Заглушка — эвристика и inpainting."""
    def remove_watermark(self, image_rgb: np.ndarray) -> np.ndarray:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        mask = detect_watermark_mask_cv(image_bgr)
        result_bgr = inpaint_image_cv(image_bgr, mask, method="telea")
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return result_rgb


def load_model_checkpoint(checkpoint_path: str) -> object:
    """Загрузка модели или fallback."""
    if LIGHTNING_AVAILABLE and WatermarkDetection is not None:
        try:
            if hasattr(WatermarkDetection, "load_from_checkpoint"):
                logger.info("Загрузка из чекпоинта: %s", checkpoint_path)
                return WatermarkDetection.load_from_checkpoint(checkpoint_path=checkpoint_path)
            else:
                logger.info("Создание модели без чекпоинта")
                return WatermarkDetection()
        except Exception as e:
            logger.warning("Не удалось загрузить чекпоинт: %s. Используем fallback. (%s)", checkpoint_path, e)
    else:
        logger.info("Модель недоступна, fallback.")
    return WatermarkDetectionFallback()


# ----------------- интерфейс Streamlit -----------------
def streamlit_app(checkpoint_path: str = "./checkpoints/best_model.ckpt"):
    st.title("Watermark Removal App")
    st.write("Загрузите изображение — приложение попытается удалить водяной знак.")
    uploaded_file = st.file_uploader("Загрузите изображение:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_arr = preprocess_image(uploaded_file)
        with st.spinner("Загрузка модели и обработка..."):
            model = load_model_checkpoint(checkpoint_path)
            try:
                cleaned = model.remove_watermark(original_arr)
            except Exception as e:
                logger.exception("Ошибка модели: %s", e)
                st.error(f"Ошибка модели: {e}\nИспользуем fallback-метод.")
                cleaned = WatermarkDetectionFallback().remove_watermark(original_arr)

        png_bytes = show_result(original_arr, cleaned)
        if png_bytes:
            st.download_button("Скачать результат", data=png_bytes, file_name="cleaned.png", mime="image/png")


# ----------------- CLI функции -----------------
def cli_single(input_path: str, output_path: str, checkpoint: Optional[str] = None):
    arr = preprocess_image(input_path)
    model = load_model_checkpoint(checkpoint or "./checkpoints/best_model.ckpt")
    try:
        cleaned = model.remove_watermark(arr)
    except Exception:
        cleaned = WatermarkDetectionFallback().remove_watermark(arr)
    result_pil = Image.fromarray(cleaned)
    ensure_parent_dir(output_path)
    result_pil.save(output_path, format="PNG")
    print("Сохранено:", output_path)


def cli_batch(input_folder: str, output_folder: str, workers: int = 4, checkpoint: Optional[str] = None):
    files = [p for p in Path(input_folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not files:
        print("Нет изображений в папке", input_folder)
        return
    model = load_model_checkpoint(checkpoint or "./checkpoints/best_model.ckpt")
    ensure_parent_dir(output_folder)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max(1, workers)) as exe:
        futures = {}
        for p in files:
            out = str(Path(output_folder) / (p.stem + "_clean.png"))
            futures[exe.submit(process_one_file, p, out, model)] = (str(p), out)
        for fut in as_completed(futures):
            src, dst = futures[fut]
            try:
                fut.result()
                print("Обработано:", src, "->", dst)
            except Exception as e:
                print("Ошибка при обработке:", src, ":", e)


def process_one_file(src_path: Path, dst_path: str, model_obj):
    arr = preprocess_image(str(src_path))
    try:
        cleaned = model_obj.remove_watermark(arr)
    except Exception:
        cleaned = WatermarkDetectionFallback().remove_watermark(arr)
    Image.fromarray(cleaned).save(dst_path, format="PNG")


def ensure_parent_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


# ----------------- точка входа -----------------
def main():
    parser = argparse.ArgumentParser(description="Watermark Removal App (Streamlit UI или CLI fallback)")
    sub = parser.add_subparsers(dest="cmd")

    p_single = sub.add_parser("single", help="Обработать одно изображение")
    p_single.add_argument("input", help="Путь к изображению")
    p_single.add_argument("output", help="Путь для сохранения результата")
    p_single.add_argument("--checkpoint", help="Путь к чекпоинту модели", default="./checkpoints/best_model.ckpt")

    p_batch = sub.add_parser("batch", help="Обработка папки")
    p_batch.add_argument("input_folder", help="Папка с изображениями")
    p_batch.add_argument("output_folder", help="Папка для сохранения результатов")
    p_batch.add_argument("--workers", type=int, default=4)
    p_batch.add_argument("--checkpoint", help="Путь к чекпоинту модели", default="./checkpoints/best_model.ckpt")

    # Если есть streamlit и нет аргументов — запустить UI
    if ST_AVAILABLE and len(sys.argv) == 1:
        streamlit_app()
        return

    args = parser.parse_args()
    if args.cmd == "single":
        cli_single(args.input, args.output, checkpoint=args.checkpoint)
    elif args.cmd == "batch":
        cli_batch(args.input_folder, args.output_folder, workers=args.workers, checkpoint=args.checkpoint)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
