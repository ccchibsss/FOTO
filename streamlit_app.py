# fixed_watermark_app.py
# Исправлённая и надёжная версия вашего приложения.
# - Работает в Streamlit (если установлен) или в CLI режиме.
# - Поддерживает загрузку реальной модели WatermarkDetection из
# lightning_module,
# если она доступна и есть чекпоинт.
# - Если модели нет, использует fallback-метод на OpenCV inpainting.
# Запуск:
# - Streamlit: streamlit run fixed_watermark_app.py
# - CLI single: python fixed_watermark_app.py single input.jpg output.png
# - CLI batch : python fixed_watermark_app.py batch /in/folder /out/folder
# --workers 4

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

# Try to import streamlit; если нет — работаем в CLI режиме
try:
    import streamlit as st  # type: ignore
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

# Try to import user lightning model (optional)
try:
    from lightning_module import WatermarkDetection  # user model
    LIGHTNING_AVAILABLE = True
except Exception:
    WatermarkDetection = None
    LIGHTNING_AVAILABLE = False

logging.basicConfig(level=logging.INFO, filename="watermark_app.log",
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------- Utilities / Fallbacks -----------------
def open_image(file_or_path) -> Image.Image:
    """Open a PIL image from a path or file-like (Streamlit uploader)."""
    if isinstance(file_or_path, str):
        return Image.open(file_or_path).convert("RGB")
    # file-like: ensure pointer at start
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
    """Heuristic mask detection used in fallback (single-channel 0/255)."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 25)
    diff = cv2.absdiff(gray, blur)
    m, s = diff.mean(), diff.std()
    thr = int(max(8, m + 0.7 * s))
    _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # remove tiny components
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = (image_cv.shape[0] * image_cv.shape[1]) * 0.0005
    out = np.zeros_like(mask)
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def inpaint_image_cv(image_cv: np.ndarray, mask_cv: np.ndarray, method: str = "telea") -> np.ndarray:
    """Inpaint masked regions. mask should be single-channel 0/255."""
    if mask_cv.ndim == 3:
        mask = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY)
    else:
        mask = mask_cv
    mask_bin = (mask > 0).astype("uint8") * 255
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    try:
        return cv2.inpaint(image_cv, mask_bin, 3, flag)
    except Exception:
        blurred = cv2.GaussianBlur(image_cv, (21, 21), 0)
        out = image_cv.copy()
        out[mask_bin == 255] = blurred[mask_bin == 255]
        return out


# ----------------- preprocess_image and show_result (original utils
# replacements) -----------------
def preprocess_image(file_or_path) -> np.ndarray:
    """
    Возвращает изображение как numpy array в RGB (H, W, C).
    Принимает путь или file-like (Streamlit uploader).
    """
    img = open_image(file_or_path)
    arr = np.array(img)  # RGB
    return arr


def show_result(original: np.ndarray, cleaned: np.ndarray, title_original: str = "Original",
                title_cleaned: str = "Cleaned"):
    """
    Показывает результат либо в Streamlit, либо сохраняет файл в CLI.
    Возвращает bytes результата (PNG) для скачивания в Streamlit.
    """
    orig_pil = Image.fromarray(original)
    clean_pil = Image.fromarray(cleaned)
    if ST_AVAILABLE:
        col1, col2 = st.columns(2)
        col1.image(orig_pil, caption=title_original, use_column_width=True)
        col2.image(clean_pil, caption=title_cleaned, use_column_width=True)
        return pil_to_bytes(clean_pil, fmt="PNG")
    else:
        # В CLI просто сохраняем временный файл и печатаем путь
        out = Path(tempfile_filename("cleaned_"))  # defined below
        clean_pil.save(out, format="PNG")
        print("Saved cleaned image to:", out)
        return None


# helper to create a temp filename
def tempfile_filename(prefix: str = "tmp_", suffix: str = ".png") -> str:
    import tempfile
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path


# ----------------- Model wrapper / fallback -----------------
class WatermarkDetectionFallback:
    """
    Простая замена модели: детектирует маску эвристикой и делает inpainting.
    Метод remove_watermark принимает numpy array RGB и возвращает RGB numpy array.
    """
    def remove_watermark(self, image_rgb: np.ndarray) -> np.ndarray:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        mask = detect_watermark_mask_cv(image_bgr)
        result_bgr = inpaint_image_cv(image_bgr, mask, method="telea")
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return result_rgb


def load_model_checkpoint(checkpoint_path: str) -> object:
    """
    Попытка загрузить модель из lightning_module.WatermarkDetection.
    Если не удалось — возвращаем fallback-объект.
    """
    if LIGHTNING_AVAILABLE and WatermarkDetection is not None:
        try:
            # Поддерживаем как класс с методом load_from_checkpoint, так и простой конструктор
            if hasattr(WatermarkDetection, "load_from_checkpoint"):
                logger.info("Loading WatermarkDetection from checkpoint: %s", checkpoint_path)
                return WatermarkDetection.load_from_checkpoint(checkpoint_path=checkpoint_path)
            else:
                logger.info("Instantiating WatermarkDetection() without checkpoint")
                return WatermarkDetection()
        except Exception as e:
            logger.warning("Failed to load WatermarkDetection checkpoint: %s. Using fallback. (%s)", checkpoint_path, e)
    else:
        logger.info("Lightning model not available, using fallback.")
    return WatermarkDetectionFallback()


# ----------------- Streamlit app -----------------
def streamlit_app(checkpoint_path: str = "./checkpoints/best_model.ckpt"):
    st.title("Watermark Removal App")
    st.write("Загрузите изображение — приложение попытается удалить водяной знак.")
    uploaded_file = st.file_uploader("Загрузите изображение:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_arr = preprocess_image(uploaded_file)  # RGB numpy
        with st.spinner("Загрузка модели и обработка..."):
            model = load_model_checkpoint(checkpoint_path)
            try:
                cleaned = model.remove_watermark(original_arr)
            except Exception as e:
                logger.exception("Model processing failed: %s", e)
                st.error(f"Ошибка при обработке модели: {e}\nПопытка fallback-обработки.")
                cleaned = WatermarkDetectionFallback().remove_watermark(original_arr)

        png_bytes = show_result(original_arr, cleaned)
        if png_bytes:
            st.download_button("Скачать результат", data=png_bytes, file_name="cleaned.png", mime="image/png")


# ----------------- CLI functionality -----------------
def cli_single(input_path: str, output_path: str, checkpoint: Optional[str] = None):
    arr = preprocess_image(input_path)
    model = load_model_checkpoint(checkpoint or "./checkpoints/best_model.ckpt")
    try:
        cleaned = model.remove_watermark(arr)
    except Exception as e:
        logger.exception("Model processing failed, using fallback: %s", e)
        cleaned = WatermarkDetectionFallback().remove_watermark(arr)
    result_pil = Image.fromarray(cleaned)
    ensure_parent_dir(output_path)
    result_pil.save(output_path, format="PNG")
    print("Saved:", output_path)


def cli_batch(input_folder: str, output_folder: str, workers: int = 4, checkpoint: Optional[str] = None):
    files = [p for p in Path(input_folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not files:
        print("No images found in", input_folder)
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
                print("Processed:", src, "->", dst)
            except Exception as e:
                print("Failed:", src, ":", e)


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


# ----------------- Entrypoint -----------------
def main():
    parser = argparse.ArgumentParser(description="Watermark Removal App (Streamlit UI or CLI fallback)")
    sub = parser.add_subparsers(dest="cmd")

    p_single = sub.add_parser("single", help="Process single image")
    p_single.add_argument("input", help="Input image path")
    p_single.add_argument("output", help="Output image path")
    p_single.add_argument("--checkpoint", help="Path to model checkpoint", default="./checkpoints/best_model.ckpt")

    p_batch = sub.add_parser("batch", help="Process folder")
    p_batch.add_argument("input_folder", help="Input folder")
    p_batch.add_argument("output_folder", help="Output folder")
    p_batch.add_argument("--workers", type=int, default=4)
    p_batch.add_argument("--checkpoint", help="Path to model checkpoint", default="./checkpoints/best_model.ckpt")

    # If Streamlit present and no CLI args, run Streamlit UI
    if ST_AVAILABLE and len(sys.argv) == 1:
        # Note: when running `streamlit run script.py`, streamlit provides args; this branch works for direct python run.
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
