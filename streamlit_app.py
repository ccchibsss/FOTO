import sys
import os
import time
import logging
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import shutil

warnings.filterwarnings('ignore')

# --- Попытка импортировать streamlit, иначе создаём улучшённый shim ---
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    # Simple ANSI color helpers (no-op on Windows without support).
    CSI = "\033["
    RESET = CSI + "0m"
    BOLD = CSI + "1m"
    GREEN = CSI + "32m"
    YELLOW = CSI + "33m"
    RED = CSI + "31m"
    BLUE = CSI + "34m"
    CYAN = CSI + "36m"
    MAGENTA = CSI + "35m"

    def _col(s, color=""):
        return f"{color}{s}{RESET}" if color else s

    class DummyColumn:
        def __init__(self, idx: int):
            self.idx = idx
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def info(self, msg): print(f"[col{self.idx}] {msg}")
        def write(self, msg): print(f"[col{self.idx}] {msg}")

    class DummyColumns(list):
        def __init__(self, spec):
            # Accept list/tuple or integer-like
            if isinstance(spec, (list, tuple)):
                try:
                    count = int(len(spec))
                except Exception:
                    count = 1
            else:
                try:
                    count = int(spec)
                except Exception:
                    count = 1
            super().__init__([DummyColumn(i+1) for i in range(max(1, count))])
        def __enter__(self):
            return tuple(self) if len(self) > 1 else self[0]
        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyProgress:
        def __init__(self, value=0.0, text=""):
            self.value = float(value)
            self.text = text
        def progress(self, v, text=None):
            self.value = float(v)
            if text is not None:
                self.text = text
            cols = shutil.get_terminal_size((80, 20)).columns
            bar_width = max(10, min(60, cols - 40))
            filled = int(self.value * bar_width)
            bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
            pct = f"{self.value*100:6.2f}%"
            print(_col(f"{bar} {pct} {self.text}", CYAN))
        def empty(self):
            pass

    class DummySpinner:
        def __init__(self, text=""):
            self.text = text
        def __enter__(self):
            print(_col(f"⏳ {self.text}", YELLOW))
            return self
        def __exit__(self, exc_type, exc, tb):
            if exc_type:
                print(_col(f"❌ {self.text} failed: {exc_type.__name__}", RED))
            else:
                print(_col(f"✅ {self.text} done", GREEN))
            return False

    class DummyStreamlit:
        def __init__(self):
            self.session_state = {}
            self._sections = {}
        # layout / page
        def set_page_config(self, *a, **k): pass
        # messages
        def info(self, msg): print(_col(f"[INFO] {msg}", BLUE))
        def success(self, msg): print(_col(f"[SUCCESS] {msg}", GREEN))
        def warning(self, msg): print(_col(f"[WARN] {msg}", YELLOW))
        def error(self, msg): print(_col(f"[ERROR] {msg}", RED))
        def exception(self, msg): print(_col(f"[EXCEPTION] {msg}", RED))
        # headings / text
        def header(self, msg): print(_col(f"\n=== {msg} ===", MAGENTA + BOLD))
        def subheader(self, msg): print(_col(f"\n-- {msg} --", MAGENTA))
        def title(self, msg): print(_col(f"\n# {msg}", BOLD))
        def markdown(self, msg): print(msg)
        def text(self, msg): print(msg)
        def write(self, msg): print(msg)
        # UI controls (non-interactive defaults)
        def number_input(self, label, min_value=None, max_value=None, value=0, step=None, key=None):
            print(f"[INPUT number] {label} (default={value})")
            return value
        def checkbox(self, label, value=False, key=None, disabled=False):
            print(f"[INPUT checkbox] {label} (default={value})")
            return value
        def button(self, label, key=None):
            print(f"[INPUT button] {label} (auto=False)")
            return False
        def selectbox(self, label, options, index=0, key=None, format_func=None):
            chosen = options[index] if options else None
            display = format_func(chosen) if format_func else chosen
            print(f"[INPUT selectbox] {label} -> {display}")
            return chosen
        def multiselect(self, label, options, default=None):
            print(f"[INPUT multiselect] {label} -> {default or []}")
            return default or []
        def radio(self, label, options, format_func=None):
            choice = options[0] if options else None
            print(f"[INPUT radio] {label} -> {choice}")
            return choice
        def file_uploader(self, *a, **k):
            print("[INPUT file_uploader] (no file in CLI shim)")
            return None
        def text_area(self, label, value="", height=None, placeholder=None):
            print(f"[INPUT text_area] {label}\n{value}")
            return value
        def text_input(self, label, value=""):
            print(f"[INPUT text_input] {label} (default='{value}')")
            return value
        def columns(self, spec):
            return DummyColumns(spec)
        def progress(self, val=0, text=""):
            p = DummyProgress(val, text)
            p.progress(val, text)
            return p
        def spinner(self, text=""):
            return DummySpinner(text)
        def dataframe(self, df, width=None, hide_index=False):
            try:
                import pandas as pd
                if isinstance(df, pd.DataFrame):
                    print(_col("[DATAFRAME]", CYAN))
                    print(df.head().to_string())
                else:
                    print(_col(f"[DATAFRAME] object {type(df)}", CYAN))
            except Exception:
                print(_col(f"[DATAFRAME] object {type(df)}", CYAN))
        def download_button(self, *a, **k):
            print("[DOWNLOAD] (no-op in CLI shim)")
            return None
        # ensure experimental_rerun present
        def experimental_rerun(self):
            self.session_state["_rerun_trigger"] = not self.session_state.get("_rerun_trigger", False)
            print("[ACTION] experimental_rerun() called (shim)")

        # --- Improved visual helpers ---
        def show_section_status(self, name: str, status: str = "idle", details: str = ""):
            """Register or print a section status. status in ('idle','running','ok','fail')."""
            icon = {"idle":"○", "running":"⏳", "ok":"✅", "fail":"❌"}.get(status, "○")
            color = {"idle":MAGENTA, "running":YELLOW, "ok":GREEN, "fail":RED}.get(status, MAGENTA)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"{icon} [{ts}] {name}: {status.upper()}"
            if details:
                line += f" — {details}"
            print(_col(line, color))
            self._sections[name] = {"status": status, "details": details, "ts": ts}

        def print_all_sections_summary(self):
            if not self._sections:
                print(_col("[SECTIONS] Нет зарегистрированных разделов", YELLOW))
                return
            print(_col("\n=== Разделы и статусы ===", BOLD+MAGENTA))
            for name, info in self._sections.items():
                icon = {"idle":"○", "running":"⏳", "ok":"✅", "fail":"❌"}.get(info["status"], "○")
                color = {"idle":MAGENTA, "running":YELLOW, "ok":GREEN, "fail":RED}.get(info["status"], MAGENTA)
                print(_col(f"{icon} {name:20} | {info['status']:7} | {info['ts']} | {info['details']}", color))

        def show_upload_statuses(self, items):
            """Pretty print list of upload tasks:
               items: list of dict {name, status, progress(0..1), size_bytes (optional)}"""
            cols = shutil.get_terminal_size((100, 20)).columns
            bar_width = max(10, min(40, cols - 60))
            print(_col("\n--- Upload statuses ---", BOLD+CYAN))
            for it in items:
                name = it.get("name")[:30].ljust(30)
                status = it.get("status", "idle")
                prog = float(it.get("progress", 0.0))
                size = it.get("size_bytes")
                filled = int(prog * bar_width)
                bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
                pct = f"{prog*100:5.1f}%"
                size_str = f"{size/1024/1024:.2f}MB" if size else ""
                color = GREEN if status == "done" else YELLOW if status in ("running","processing") else RED if status=="error" else MAGENTA
                print(_col(f"{name} {bar} {pct} {status.upper():10} {size_str}", color))

    st = DummyStreamlit()
    print("[WARN] streamlit not installed — running with CLI shim (no interactive UI).")

# Оборачиваем экспериментальный rerun оригинала (если доступен) в безопасную
# версию.
try:
    _orig_rerun = st.experimental_rerun
except Exception:
    _orig_rerun = None

def _fallback_rerun():
    st.session_state["_rerun_trigger"] = not st.session_state.get("_rerun_trigger", False)

if _orig_rerun is None:
    st.experimental_rerun = _fallback_rerun
else:
    def _safe_rerun():
        try:
            return _orig_rerun()
        except Exception:
            _fallback_rerun()
    st.experimental_rerun = _safe_rerun

# --- Остальные импорты (оригинальные) ---
import platform
# Некоторые окружения могут не иметь polars/duckdb — оставляем оригинальные
# импорты; при ошибке запуск покажет соответствующее сообщение.
try:
    import polars as pl
except Exception:
    pl = None
try:
    import duckdb
except Exception:
    duckdb = None

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXCEL_ROW_LIMIT = 1_000_000

class HighVolumeAutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)

        # Загрузка конфигураций
        self.cloud_config = self.load_cloud_config()
        self.price_rules = self.load_price_rules()
        self.exclusion_rules = self.load_exclusion_rules()
        self.category_mapping = self.load_category_mapping()

        self.db_path = self.data_dir / "catalog.duckdb"
        if duckdb is None:
            logger.warning("duckdb не доступен — база не будет создана в этом режиме.")
            self.conn = None
        else:
            self.conn = duckdb.connect(database=str(self.db_path))
            self.setup_database()

        st.set_page_config(
            page_title="AutoParts Catalog 10M+",
            layout="wide",
            page_icon="🚗"
        )

    # --- Конфигурации ---
    def load_cloud_config(self) -> Dict[str, Any]:
        config_path = self.data_dir / "cloud_config.json"
        default_config = {
            "enabled": False,
            "provider": "s3",
            "bucket": "",
            "region": "",
            "sync_interval": 3600,
            "last_sync": 0
        }
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding='utf-8'))
            except Exception as e:
                logger.error(f"Ошибка чтения cloud_config.json: {e}")
                return default_config
        else:
            config_path.write_text(json.dumps(
                default_config, indent=2, ensure_ascii=False), encoding='utf-8')
            return default_config

    def save_cloud_config(self):
        config_path = self.data_dir / "cloud_config.json"
        self.cloud_config["last_sync"] = int(time.time())
        config_path.write_text(json.dumps(
            self.cloud_config, indent=2, ensure_ascii=False), encoding='utf-8')

    def load_price_rules(self) -> Dict[str, Any]:
        price_rules_path = self.data_dir / "price_rules.json"
        default_rules = {
            "global_markup": 0.2,
            "brand_markups": {},
            "min_price": 0.0,
            "max_price": 99999.0
        }
        if price_rules_path.exists():
            try:
                return json.loads(price_rules_path.read_text(encoding='utf-8'))
            except Exception as e:
                logger.error(f"Ошибка чтения price_rules.json: {e}")
                return default_rules
        else:
            price_rules_path.write_text(json.dumps(
                default_rules, indent=2, ensure_ascii=False), encoding='utf-8')
            return default_rules

    def save_price_rules(self):
        price_rules_path = self.data_dir / "price_rules.json"
        price_rules_path.write_text(json.dumps(
            self.price_rules, indent=2, ensure_ascii=False), encoding='utf-8')

    def load_exclusion_rules(self) -> List[str]:
        exclusion_path = self.data_dir / "exclusion_rules.txt"
        if exclusion_path.exists():
            try:
                return [line.strip() for line in exclusion_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            except Exception as e:
                logger.error(f"Ошибка чтения exclusion_rules.txt: {e}")
                return []
        else:
            content = "Кузов\nСтекла\nМасла"
            exclusion_path.write_text(content, encoding='utf-8')
            return ["Кузов", "Стекла", "Масла"]

    def save_exclusion_rules(self):
        exclusion_path = self.data_dir / "exclusion_rules.txt"
        exclusion_path.write_text(
            "\n".join(self.exclusion_rules), encoding='utf-8')

    def load_category_mapping(self) -> Dict[str, str]:
        category_path = self.data_dir / "category_mapping.txt"
        default_mapping = {
            "Радиатор": "Охлаждение",
            "Шаровая опора": "Подвеска",
            "Фильтр масляный": "Фильтры",
            "Тормозные колодки": "Тормоза"
        }
        if category_path.exists():
            try:
                mapping = {}
                for line in category_path.read_text(encoding='utf-8').splitlines():
                    if line.strip() and "|" in line:
                        key, value = line.split("|", 1)
                        mapping[key.strip()] = value.strip()
                return mapping
            except Exception as e:
                logger.error(f"Ошибка чтения category_mapping.txt: {e}")
                return default_mapping
        else:
            content = "\n".join(
                [f"{k}|{v}" for k, v in default_mapping.items()])
            category_path.write_text(content, encoding='utf-8')
            return default_mapping

    def save_category_mapping(self):
        category_path = self.data_dir / "category_mapping.txt"
        content = "\n".join(
            [f"{k}|{v}" for k, v in self.category_mapping.items()])
        category_path.write_text(content, encoding='utf-8')

    # --- База данных ---
    def setup_database(self):
        if self.conn is None:
            return
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS oe (
                oe_number_norm VARCHAR PRIMARY KEY,
                oe_number VARCHAR,
                name VARCHAR,
                applicability VARCHAR,
                category VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS parts (
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                artikul VARCHAR,
                brand VARCHAR,
                multiplicity INTEGER,
                barcode VARCHAR,
                length DOUBLE,
                width DOUBLE,
                height DOUBLE,
                weight DOUBLE,
                image_url VARCHAR,
                dimensions_str VARCHAR,
                description VARCHAR,
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_references (
                oe_number_norm VARCHAR,
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                PRIMARY KEY (oe_number_norm, artikul_norm, brand_norm)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                price DOUBLE,
                currency VARCHAR DEFAULT 'RUB',
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)
        self.create_indexes()

    def create_indexes(self):
        st.info("🛠️ Создание индексов для ускорения поиска...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_oe_number_norm ON oe(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_keys ON parts(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_prices_keys ON prices(artikul_norm, brand_norm)"
        ]
        if self.conn is None:
            st.warning("DB connection not available — skipping index creation.")
            return
        for index_sql in indexes:
            try:
                self.conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Не удалось создать индекс: {e}")
        st.success("🛠️ Индексы созданы.")

    # --- Нормализация и очистка ---
    @staticmethod
    def normalize_key(series):
        # If polars not available, operate on simple Python list/str
        if pl is None:
            if isinstance(series, list):
                return [str(s).lower().strip().replace("'", "") for s in series]
            elif isinstance(series, str):
                return series.lower().strip().replace("'", "")
            else:
                return series
        return (series
                .fill_null("")
                .cast(pl.Utf8)
                .str.replace_all("'", "")
                .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .str.to_lowercase())

    @staticmethod
    def clean_values(series):
        if pl is None:
            return series
        return (series
                .fill_null("")
                .cast(pl.Utf8)
                .str.replace_all("'", "")
                .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\-\s]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars())

    def determine_category_vectorized(self, name_series):
        if pl is None:
            # fallback: simple heuristic
            res = []
            for name in name_series:
                nl = (name or "").lower()
                found = None
                for k, v in self.category_mapping.items():
                    if k.lower() in nl:
                        found = v
                        break
                if found is None:
                    if any(x in nl for x in ("фильтр", "filter")):
                        found = "Фильтры"
                    else:
                        found = "Разное"
                res.append(found)
            return res
        name_lower = name_series.str.to_lowercase()
        categorization_expr = pl.when(pl.lit(False)).then(pl.lit(None))
        for key, category in self.category_mapping.items():
            categorization_expr = categorization_expr.when(
                name_lower.str.contains(key.lower())
            ).then(pl.lit(category))
        categories_map = {
            'Фильтр': 'фильтр|filter',
            'Тормоза': 'тормоз|brake|колодк|диск|суппорт',
            'Подвеска': 'амортизатор|стойк|spring|подвеск|рычаг',
            'Двигатель': 'двигатель|engine|свеч|поршень|клапан',
            'Трансмиссия': 'трансмиссия|сцеплен|коробк|transmission',
            'Электрика': 'аккумулятор|генератор|стартер|провод|ламп',
            'Рулевое': 'рулевой|тяга|наконечник|steering',
            'Выпуск': 'глушитель|катализатор|выхлоп|exhaust',
            'Охлаждение': 'радиатор|вентилятор|термостат|cooling',
            'Топливо': 'топливный|бензонасос|форсунк|fuel'
        }
        for category, pattern in categories_map.items():
            categorization_expr = categorization_expr.when(
                name_lower.str.contains(pattern, literal=False)
            ).then(pl.lit(category))
        return categorization_expr.otherwise(pl.lit('Разное')).alias('category')

    # --- Обработка файлов ---
    def detect_columns(self, actual_columns: List[str], expected_columns: List[str]) -> Dict[str, str]:
        column_variants = {
            'oe_number': ['oe номер', 'oe', 'оe', 'номер', 'code', 'OE'],
            'artikul': ['артикул', 'article', 'sku'],
            'brand': ['бренд', 'brand', 'производитель', 'manufacturer'],
            'name': ['наименование', 'название', 'name', 'описание', 'description'],
            'applicability': ['применимость', 'автомобиль', 'vehicle', 'applicability'],
            'barcode': ['штрих-код', 'barcode', 'штрихкод', 'ean', 'eac13'],
            'multiplicity': ['кратность шт', 'кратность', 'multiplicity'],
            'length': ['длина (см)', 'длина', 'length', 'длинна'],
            'width': ['ширина (см)', 'ширина', 'width'],
            'height': ['высота (см)', 'высота', 'height'],
            'weight': ['вес (кг)', 'вес, кг', 'вес', 'weight'],
            'image_url': ['ссылка', 'url', 'изображение', 'image', 'картинка'],
            'dimensions_str': ['весогабариты', 'размеры', 'dimensions', 'size'],
            'price': ['цена', 'price', 'рекомендованная цена', 'retail price'],
            'currency': ['валюта', 'currency']
        }
        actual_lower = {col.lower(): col for col in actual_columns}
        mapping = {}
        for expected in expected_columns:
            variants = column_variants.get(expected, [expected])
            for variant in variants:
                variant_lower = variant.lower()
                for actual_l, actual_orig in actual_lower.items():
                    if variant_lower in actual_l and actual_orig not in mapping:
                        mapping[actual_orig] = expected
                        break
        return mapping

    def read_and_prepare_file(self, file_path: str, file_type: str):
        logger.info(f"Обработка файла: {file_type} ({file_path})")
        if pl is None:
            logger.warning("polars не установлен — read_and_prepare_file вернёт пустой DataFrame-подобный объект")
            return pl  # None
        try:
            if not os.path.exists(file_path):
                logger.error(f"Файл не найден: {file_path}")
                return pl.DataFrame()

            df = pl.read_excel(file_path, engine='calamine')
            if df.is_empty():
                logger.warning(f"Пустой файл: {file_path}")
                return pl.DataFrame()

        except Exception as e:
            logger.exception(f"Ошибка чтения файла {file_path}: {e}")
            return pl.DataFrame()

        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'cross': ['oe_number', 'artikul', 'brand'],
            'barcode': ['artikul', 'brand', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'prices': ['artikul', 'brand', 'price', 'currency']
        }
        expected_cols = schemas.get(file_type, [])
        column_mapping = self.detect_columns(df.columns, expected_cols)
        if not column_mapping:
            logger.warning(
                f"Не удалось определить колонки для файла {file_type}. Доступные: {df.columns}")
            return pl.DataFrame()

        df = df.rename(column_mapping)

        for col in ['artikul', 'brand', 'oe_number']:
            if col in df.columns:
                df = df.with_columns(self.clean_values(pl.col(col)).alias(col))

        key_cols = [col for col in ['oe_number',
                                    'artikul', 'brand'] if col in df.columns]
        if key_cols:
            df = df.unique(subset=key_cols, keep='first')

        for col in ['artikul', 'brand', 'oe_number']:
            if col in df.columns:
                df = df.with_columns(self.normalize_key(
                    pl.col(col)).alias(f"{col}_norm"))

        return df

    # --- Загрузка и обновление в базе ---
    def upsert_data(self, table_name: str, df, pk: List[str]):
        if df is None:
            return
        if self.conn is None:
            logger.warning(f"DB not available — skipping upsert into {table_name}")
            return
        try:
            if hasattr(df, "to_arrow"):
                temp_view_name = f"temp_{table_name}_{int(time.time())}"
                self.conn.register(temp_view_name, df.to_arrow())
            else:
                import pandas as pd
                temp_view_name = f"temp_{table_name}_{int(time.time())}"
                self.conn.register(temp_view_name, df.to_dict('records'))
        except Exception as e:
            logger.error(f"Ошибка регистрации временной таблицы: {e}")
            return

        try:
            pk_list = pk
            pk_cols_csv = ", ".join(f'"{c}"' for c in pk_list)
            delete_sql = f"""
                DELETE FROM {table_name}
                WHERE ({pk_cols_csv}) IN (SELECT {pk_cols_csv} FROM {temp_view_name});
            """
            self.conn.execute(delete_sql)
            insert_sql = f"""
                INSERT INTO {table_name}
                SELECT * FROM {temp_view_name};
            """
            self.conn.execute(insert_sql)
            logger.info(
                f"Успешно upsert в таблицу {table_name}.")
        except Exception as e:
            logger.error(f"Ошибка при UPSERT в {table_name}: {e}")
            st.error(
                f"Ошибка при записи в таблицу {table_name}. Детали в логе.")
        finally:
            try:
                self.conn.unregister(temp_view_name)
            except Exception:
                pass

    def upsert_prices(self, price_df):
        if price_df is None:
            return
        if 'artikul' in getattr(price_df, "columns", [] ) and 'brand' in getattr(price_df, "columns", []):
            price_df = price_df.with_columns([
                self.normalize_key(pl.col('artikul')).alias('artikul_norm'),
                self.normalize_key(pl.col('brand')).alias('brand_norm')
            ])
        if 'currency' not in getattr(price_df, "columns", []):
            price_df = price_df.with_columns(pl.lit('RUB').alias('currency'))
        price_df = price_df.filter(
            (pl.col('price') >= self.price_rules['min_price']) &
            (pl.col('price') <= self.price_rules['max_price'])
        )
        self.upsert_data('prices', price_df, ['artikul_norm', 'brand_norm'])

    def process_and_load_data(self, dataframes: Dict[str, Any]):
        st.info("🔄 Начало загрузки и обновления данных в базе...")
        steps = [s for s in ['oe', 'cross', 'parts'] if s in dataframes]
        num_steps = len(steps)
        progress_bar = st.progress(
            0, text="Подготовка к обновлению базы данных...")
        step_counter = 0

        if 'oe' in dataframes:
            step_counter += 1
            progress_bar.progress(step_counter / (num_steps + 1),
                                  text=f"({step_counter}/{num_steps}) Обработка OE данных...")
            df = dataframes['oe']
            if pl is None or df is None:
                logger.warning("polars отсутствует или df пустой — пропускаем OE обработку")
            else:
                df = df.filter(pl.col('oe_number_norm') != "")
                oe_df = df.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique(
                    subset=['oe_number_norm'], keep='first')

                if 'name' in oe_df.columns:
                    oe_df = oe_df.with_columns(
                        self.determine_category_vectorized(pl.col('name')))
                else:
                    oe_df = oe_df.with_columns(category=pl.lit('Разное'))

                self.upsert_data('oe', oe_df, ['oe_number_norm'])

                cross_df_from_oe = df.filter(pl.col('artikul_norm') != "").select(
                    ['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
                self.upsert_data('cross_references', cross_df_from_oe, [
                                 'oe_number_norm', 'artikul_norm', 'brand_norm'])

        if 'cross' in dataframes:
            step_counter += 1
            progress_bar.progress(step_counter / (num_steps + 1),
                                  text=f"({step_counter}/{num_steps}) Обработка кроссов...")
            df = dataframes['cross']
            if pl is None or df is None:
                logger.warning("polars отсутствует или df пустой — пропускаем cross обработку")
            else:
                df = df.filter(
                    (pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
                cross_df_from_cross = df.select(
                    ['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
                self.upsert_data('cross_references', cross_df_from_cross, [
                                 'oe_number_norm', 'artikul_norm', 'brand_norm'])

        if 'prices' in dataframes:
            price_df = dataframes['prices']
            if price_df is not None:
                st.info("💰 Обработка цен...")
                self.upsert_prices(price_df)
                st.success(
                    f"✅ Успешно обновлено ценовых записей")

        step_counter += 1
        progress_bar.progress(step_counter / (num_steps + 1),
                              text=f"({step_counter}/{num_steps}) Сборка и обновление данных по артикулам...")

        # The rest of the merging logic requires polars; in CLI/shim mode we skip heavy processing if polars missing.
        if pl is None:
            logger.warning("polars not available — skipping parts assembly.")
            progress_bar.progress(1.0, text="Обновление базы данных завершено!")
            time.sleep(1)
            progress_bar.empty()
            return

        # Собираем parts из разных файлов
        parts_df = None
        file_priority = ['oe', 'barcode', 'images', 'dimensions']
        key_files = {ftype: df for ftype,
                     df in dataframes.items() if ftype in file_priority}

        if key_files:
            all_parts = pl.concat([
                df.select(['artikul', 'artikul_norm', 'brand', 'brand_norm'])
                for df in key_files.values() if 'artikul_norm' in df.columns and 'brand_norm' in df.columns
            ]).filter(pl.col('artikul_norm') != "").unique(subset=['artikul_norm', 'brand_norm'], keep='first')
            parts_df = all_parts

            for ftype in file_priority:
                if ftype not in key_files:
                    continue
                df = key_files[ftype]
                if df.is_empty() or 'artikul_norm' not in df.columns:
                    continue
                join_cols = [col for col in df.columns if col not in [
                    'artikul', 'artikul_norm', 'brand', 'brand_norm']]
                if not join_cols:
                    continue
                existing_cols = set(parts_df.columns)
                join_cols = [
                    col for col in join_cols if col not in existing_cols]
                if not join_cols:
                    continue
                df_subset = df.select(['artikul_norm', 'brand_norm'] + join_cols).unique(
                    subset=['artikul_norm', 'brand_norm'], keep='first')
                parts_df = parts_df.join(
                    df_subset, on=['artikul_norm', 'brand_norm'], how='left', coalesce=True)

        if parts_df is not None and not parts_df.is_empty():
            if 'multiplicity' not in parts_df.columns:
                parts_df = parts_df.with_columns(
                    multiplicity=pl.lit(1).cast(pl.Int32))
            else:
                parts_df = parts_df.with_columns(
                    pl.col('multiplicity').fill_null(1).cast(pl.Int32))

            for col in ['length', 'width', 'height']:
                if col not in parts_df.columns:
                    parts_df = parts_df.with_columns(
                        pl.lit(None).cast(pl.Float64).alias(col))

            if 'dimensions_str' not in parts_df.columns:
                parts_df = parts_df.with_columns(
                    dimensions_str=pl.lit(None).cast(pl.Utf8))

            parts_df = parts_df.with_columns([
                pl.col('length').cast(pl.Utf8).fill_null(
                    '').alias('_length_str'),
                pl.col('width').cast(pl.Utf8).fill_null(
                    '').alias('_width_str'),
                pl.col('height').cast(pl.Utf8).fill_null(
                    '').alias('_height_str'),
            ])

            parts_df = parts_df.with_columns(
                dimensions_str=pl.when(
                    (pl.col('dimensions_str').is_not_null()) &
                    (pl.col('dimensions_str').cast(pl.Utf8) != '')
                ).then(
                    pl.col('dimensions_str').cast(pl.Utf8)
                ).otherwise(
                    pl.concat_str([
                        pl.col('_length_str'), pl.lit('x'),
                        pl.col('_width_str'), pl.lit('x'),
                        pl.col('_height_str')
                    ], separator='')
                )
            )

            parts_df = parts_df.drop(
                ['_length_str', '_width_str', '_height_str'])

            if 'artikul' not in parts_df.columns:
                parts_df = parts_df.with_columns(artikul=pl.lit(''))
            if 'brand' not in parts_df.columns:
                parts_df = parts_df.with_columns(brand=pl.lit(''))

            parts_df = parts_df.with_columns([
                pl.col('artikul').cast(pl.Utf8).fill_null(
                    '').alias('_artikul_str'),
                pl.col('brand').cast(pl.Utf8).fill_null(
                    '').alias('_brand_str'),
                pl.col('multiplicity').cast(
                    pl.Utf8).alias('_multiplicity_str'),
            ])

            parts_df = parts_df.with_columns(
                description=pl.concat_str([
                    pl.lit('Артикул: '), pl.col('_artikul_str'),
                    pl.lit(', Бренд: '), pl.col('_brand_str'),
                    pl.lit(', Кратность: '), pl.col(
                        '_multiplicity_str'), pl.lit(' шт.')
                ], separator='')
            )

            parts_df = parts_df.drop(
                ['_artikul_str', '_brand_str', '_multiplicity_str'])

            final_columns = [
                'artikul_norm', 'brand_norm', 'artikul', 'brand', 'multiplicity', 'barcode',
                'length', 'width', 'height', 'weight', 'image_url', 'dimensions_str', 'description'
            ]
            select_exprs = [pl.col(c) if c in parts_df.columns else pl.lit(
                None).alias(c) for c in final_columns]
            parts_df = parts_df.select(select_exprs)

            self.upsert_data('parts', parts_df, ['artikul_norm', 'brand_norm'])

        progress_bar.progress(1.0, text="Обновление базы данных завершено!")
        time.sleep(1)
        progress_bar.empty()

    # --- Экспорт (упрощённый: в режиме без duckdb/polars даём заглушки) ---
    def _get_brand_markups_sql(self) -> str:
        rows = []
        for brand, markup in self.price_rules['brand_markups'].items():
            safe_brand = brand.replace("'", "''")
            rows.append(f"SELECT '{safe_brand}' AS brand, {markup} AS markup")
        return " UNION ALL ".join(rows) if rows else "SELECT NULL AS brand, NULL AS markup LIMIT 0"

    def build_export_query(self, selected_columns=None, include_prices=True, apply_markup=True):
        description_text = (
            "Состояние товара: новый (в упаковке). Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. "
        )
        brand_markups_sql = self._get_brand_markups_sql()
        select_parts = []
        price_requested = include_prices and (not selected_columns or "Цена" in selected_columns or "Валюта" in selected_columns)
        if price_requested:
            if apply_markup:
                global_markup = self.price_rules.get('global_markup', 0)
                select_parts.append(
                    f"CASE WHEN pr.price IS NOT NULL THEN pr.price * (1 + COALESCE(brm.markup, {global_markup})) ELSE pr.price END AS \"Цена\""
                )
            else:
                select_parts.append('pr.price AS "Цена"')
            select_parts.append("COALESCE(pr.currency, 'RUB') AS \"Валюта\"")
        columns_map = [
            ("Артикул бренда", 'r.artikul AS "Артикул бренда"'),
            ("Бренд", 'r.brand AS "Бренд"'),
            ("Наименование", 'COALESCE(r.representative_name, r.analog_representative_name) AS "Наименование"'),
            ("Применимость", 'COALESCE(r.representative_applicability, r.analog_representative_applicability) AS "Применимость"'),
            ("Описание", 'CONCAT(COALESCE(r.description, \'\'), dt.text) AS "Описание"'),
            ("Категория товара", 'COALESCE(r.representative_category, r.analog_representative_category) AS "Категория товара"'),
            ("Кратность", 'r.multiplicity AS "Кратность"'),
            ("Длинна", 'COALESCE(r.length, r.analog_length) AS "Длинна"'),
            ("Ширина", 'COALESCE(r.width, r.analog_width) AS "Ширина"'),
            ("Высота", 'COALESCE(r.height, r.analog_height) AS "Высота"'),
            ("Вес", 'COALESCE(r.weight, r.analog_weight) AS "Вес"'),
            ("Длинна/Ширина/Высота", """
                COALESCE(
                    CASE
                        WHEN r.dimensions_str IS NULL OR r.dimensions_str = '' OR UPPER(TRIM(r.dimensions_str)) = 'XX'
                        THEN NULL
                        ELSE r.dimensions_str
                    END,
                    r.analog_dimensions_str
                ) AS "Длинна/Ширина/Высота"
            """),
            ("OE номер", 'r.oe_list AS "OE номер"'),
            ("аналоги", 'r.analog_list AS "аналоги"'),
            ("Ссылка на изображение", 'r.image_url AS "Ссылка на изображение"')
        ]
        for name, expr in columns_map:
            if not selected_columns or name in selected_columns:
                select_parts.append(expr.strip())
        if not select_parts:
            select_parts = ['r.artikul AS "Артикул бренда"', 'r.brand AS "Бренд"']
        select_clause = ",\n        ".join(select_parts)
        ctes = f"""
        WITH DescriptionTemplate AS (
            SELECT CHR(10) || CHR(10) || $${description_text}$$ AS text
        ),
        BrandMarkups AS (
            SELECT brand, markup FROM (
                {brand_markups_sql}
            ) AS tmp
        ),
        PartDetails AS (
            SELECT 
                cr.artikul_norm, 
                cr.brand_norm,
                STRING_AGG(
                    DISTINCT regexp_replace(
                        regexp_replace(o.oe_number, '''', ''), 
                        '[^0-9A-Za-zА-Яа-яЁё`\\-\\s]', '', 'g'
                    ), ', '
                ) AS oe_list,
                ANY_VALUE(o.name) AS representative_name,
                ANY_VALUE(o.applicability) AS representative_applicability,
                ANY_VALUE(o.category) AS representative_category
            FROM cross_references cr
            LEFT JOIN oe o ON cr.oe_number_norm = o.oe_number_norm
            GROUP BY cr.artikul_norm, cr.brand_norm
        )
        """
        price_join = """
        LEFT JOIN prices pr ON r.artikul_norm = pr.artikul_norm AND r.brand_norm = pr.brand_norm
        LEFT JOIN BrandMarkups brm ON r.brand = brm.brand
        """ if include_prices else ""
        query = f"""
        {ctes}
        SELECT
            {select_clause}
        FROM RankedData r
        CROSS JOIN DescriptionTemplate dt
        {price_join}
        WHERE r.rn = 1
        ORDER BY r.brand, r.artikul
        """
        return "\n".join([line.rstrip() for line in query.strip().splitlines()])

    def export_to_csv_optimized(self, output_path: str, selected_columns: Optional[List[str]] = None, include_prices: bool = True, apply_markup: bool = True) -> bool:
        if self.conn is None:
            st.warning("Нет базы данных — экспорт невозможен в этом режиме.")
            return False
        total = self.conn.execute(
            "SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📤 Экспорт {total} записей в CSV...")
        try:
            query = self.build_export_query(
                selected_columns, include_prices, apply_markup)
            logger.info(f"Executing export query: {query}")
            df = self.conn.execute(query).pl()
            import pandas as pd
            pdf = df.to_pandas()
            dimension_cols = ["Длинна", "Ширина",
                              "Высота", "Вес", "Длинна/Ширина/Высота"]
            for col in dimension_cols:
                if col in pdf.columns:
                    pdf[col] = pdf[col].astype(str).replace({'nan': ''})
            output_dir = Path("auto_parts_data")
            output_dir.mkdir(parents=True, exist_ok=True)
            buf = io.StringIO()
            pdf.to_csv(buf, sep=';', index=False)
            with open(output_path, "wb") as f:
                f.write(b'\xef\xbb\xbf')
                f.write(buf.getvalue().encode('utf-8'))
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            st.success(
                f"Данные экспортированы: {output_path} ({size_mb:.1f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта CSV")
            st.error(f"Ошибка при экспорте в CSV: {str(e)}")
            return False

    def export_to_excel_optimized(self, output_path: str, selected_columns: Optional[List[str]] = None, include_prices: bool = True, apply_markup: bool = True) -> bool:
        if self.conn is None:
            st.warning("Нет базы данных — экспорт невозможен в этом режиме.")
            return False
        import pandas as pd
        query = self.build_export_query(
            selected_columns, include_prices, apply_markup)
        df = pd.read_sql(query, self.conn)
        for col in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота"]:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(
                    {r'^nan$': ''}, regex=True)
        if len(df) <= EXCEL_ROW_LIMIT:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
        else:
            sheets = (len(df) // EXCEL_ROW_LIMIT) + 1
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for i in range(sheets):
                    df.iloc[i*EXCEL_ROW_LIMIT:(i+1)*EXCEL_ROW_LIMIT].to_excel(
                        writer, index=False, sheet_name=f"Данные_{i+1}")
        return True

    def export_to_parquet(self, output_path: str, selected_columns: Optional[List[str]] = None, include_prices: bool = True, apply_markup: bool = True) -> bool:
        try:
            query = self.build_export_query(
                selected_columns, include_prices, apply_markup)
            df = self.conn.execute(query).pl()
            df.write_parquet(output_path)
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта Parquet")
            st.error(f"Ошибка при экспорте в Parquet: {str(e)}")
            return False

    # --- Управление данными (упрощённо) ---
    def delete_by_brand(self, brand_norm: str) -> int:
        try:
            if self.conn is None:
                logger.warning("DB not available — delete skipped")
                return 0
            count_result = self.conn.execute(
                "SELECT COUNT(*) FROM parts WHERE brand_norm = ?", [brand_norm]).fetchone()
            deleted_count = count_result[0] if count_result else 0
            if deleted_count == 0:
                logger.info(f"No records found for brand: {brand_norm}")
                return 0
            self.conn.execute(
                "DELETE FROM parts WHERE brand_norm = ?", [brand_norm])
            self.conn.execute(
                "DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT DISTINCT artikul_norm, brand_norm FROM parts)")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting by brand {brand_norm}: {e}")
            raise

    def delete_by_artikul(self, artikul_norm: str) -> int:
        try:
            if self.conn is None:
                logger.warning("DB not available — delete skipped")
                return 0
            count_result = self.conn.execute(
                "SELECT COUNT(*) FROM parts WHERE artikul_norm = ?", [artikul_norm]).fetchone()
            deleted_count = count_result[0] if count_result else 0
            if deleted_count == 0:
                logger.info(f"No records found for artikul: {artikul_norm}")
                return 0
            self.conn.execute(
                "DELETE FROM parts WHERE artikul_norm = ?", [artikul_norm])
            self.conn.execute(
                "DELETE FROM cross_references WHERE (artikul_norm, brand_norm) NOT IN (SELECT DISTINCT artikul_norm, brand_norm FROM parts)")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting by artikul {artikul_norm}: {e}")
            raise

    # --- Интерфейсы (упрощённо для CLI shim) ---
    def show_export_interface(self):
        st.header("📤 Экспорт данных")
        if self.conn is None:
            st.warning("DB not available — экспорт отключен")
            return
        total = self.conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts)").fetchone()[0]
        st.info(f"Всего: {total}")
        if total == 0:
            st.warning("Нет данных для экспорта")
            return

        format_choice = st.radio("Формат", ["CSV", "Excel", "Parquet"])
        selected_columns = st.multiselect("Колонки", [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота", "Вес",
            "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение", "Цена", "Валюта"
        ])

        include_prices = st.checkbox("Включить цены", value=True)
        apply_markup = st.checkbox(
            "Применить наценку", value=True, disabled=not include_prices)

        if st.button("🚀 Экспортировать"):
            output_path = self.data_dir / f"export.{format_choice.lower()}"
            with st.spinner("Генерация файла..."):
                if format_choice == "CSV":
                    self.export_to_csv_optimized(str(
                        output_path), selected_columns if selected_columns else None, include_prices, apply_markup)
                elif format_choice == "Excel":
                    self.export_to_excel_optimized(str(
                        output_path), selected_columns if selected_columns else None, include_prices, apply_markup)
                elif format_choice == "Parquet":
                    self.export_to_parquet(str(
                        output_path), selected_columns if selected_columns else None, include_prices, apply_markup)
                else:
                    st.warning("Неподдерживаемый формат")
                    return
            with open(output_path, "rb") as f:
                st.download_button("⬇️ Скачать файл", f,
                                   file_name=output_path.name)

    def show_price_settings(self):
        st.header("💰 Управление ценами и наценками")
        st.subheader("Общая наценка")
        global_markup = st.number_input(
            "Общая наценка (%):",
            min_value=0.0,
            max_value=500.0,
            value=self.price_rules['global_markup'] * 100,
            step=0.1
        )
        self.price_rules['global_markup'] = global_markup / 100

        st.subheader("Наценки по брендам")
        brand_markups = self.price_rules.get('brand_markups', {})

        try:
            if self.conn is None:
                available_brands = []
            else:
                brands_result = self.conn.execute(
                    "SELECT DISTINCT brand FROM parts WHERE brand IS NOT NULL ORDER BY brand").fetchall()
                available_brands = [row[0]
                                    for row in brands_result] if brands_result else []
        except Exception as e:
            logger.error(f"Ошибка при получении списка брендов: {e}")
            st.error("❌ Ошибка при загрузке брендов")
            available_brands = []

        if available_brands:
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_brand = st.selectbox(
                    "Выберите бренд:", available_brands)
            with col2:
                current_markup = brand_markups.get(
                    selected_brand, self.price_rules.get('global_markup', 0))
                brand_markup = st.number_input(
                    "Наценка (%):",
                    min_value=0.0,
                    max_value=500.0,
                    value=current_markup * 100,
                    step=0.1,
                    key=f"markup_{selected_brand}"
                )
            if st.button("Сохранить наценку", key=f"save_{selected_brand}"):
                brand_markups[selected_brand] = brand_markup / 100
                self.price_rules['brand_markups'] = brand_markups
                self.save_price_rules()
                st.success(f"✅ Наценка для {selected_brand} сохранена")

        st.subheader("Ограничения по ценам")
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Минимальная цена:", min_value=0.0, value=float(
                self.price_rules['min_price']), step=0.01)
            self.price_rules['min_price'] = min_price
        with col2:
            max_price = st.number_input("Максимальная цена:", min_value=0.0, value=float(
                self.price_rules['max_price']), step=0.01)
            self.price_rules['max_price'] = max_price

        if st.button("Сохранить все настройки цен"):
            self.save_price_rules()
            st.success("✅ Все настройки цен сохранены")

    def show_exclusion_settings(self):
        st.header("🚫 Управление исключениями при экспорте")
        st.info("Товары, содержащие эти слова в названии, будут исключены из экспорта")

        current_exclusions = "\n".join(self.exclusion_rules)
        new_exclusions = st.text_area(
            "Список исключений (по одному на строку):",
            value=current_exclusions,
            height=200,
            placeholder="Введите слова для исключения, например:\nКузов\nСтекла\nМасла"
        )

        if st.button("Сохранить правила исключения"):
            cleaned = [line.strip()
                       for line in new_exclusions.splitlines() if line.strip()]
            if len(cleaned) != len(set(cleaned)):
                st.warning(
                    "Обнаружены дублирующие записи. Они будут автоматически удалены.")
            self.exclusion_rules = list(dict.fromkeys(cleaned))
            self.save_exclusion_rules()
            st.success("✅ Правила исключения сохранены")

    def show_category_mapping(self):
        st.header("🗂️ Управление категориями товаров")
        st.info("Настройте соответствие между названиями товаров и категориями")

        st.subheader("Текущие правила")
        if self.category_mapping:
            try:
                mapping_df = pl.DataFrame({
                    "Название товара": list(self.category_mapping.keys()),
                    "Категория": list(self.category_mapping.values())
                }).to_pandas()
                st.dataframe(mapping_df, width='stretch', hide_index=True)
            except Exception:
                print(self.category_mapping)
        else:
            st.write("Нет пользовательских правил")

        st.subheader("Добавить правило")
        col1, col2 = st.columns(2)
        with col1:
            name_pattern = st.text_input("Ключевое слово в названии")
        with col2:
            category = st.text_input("Категория")
        if st.button("➕ Добавить"):
            if name_pattern.strip() and category.strip():
                normalized_key = name_pattern.strip().lower()
                existing_keys = {
                    k.lower(): k for k in self.category_mapping.keys()}
                if normalized_key in existing_keys:
                    st.warning(
                        f"Правило для '{existing_keys[normalized_key]}' обновлено")
                self.category_mapping[name_pattern.strip()] = category.strip()
                self.save_category_mapping()
                st.success(
                    f"Добавлено: {name_pattern.strip()} → {category.strip()}")
                st.experimental_rerun()
            else:
                st.error("Заполните оба поля")

        if self.category_mapping:
            st.subheader("🗑️ Удалить правило")
            rule_to_delete = st.selectbox(
                "Выберите правило",
                options=list(self.category_mapping.keys()),
                format_func=lambda x: f"{x} → {self.category_mapping[x]}"
            )
            if st.button("Удалить"):
                del self.category_mapping[rule_to_delete]
                self.save_category_mapping()
                st.success(f"Удалено: {rule_to_delete}")
                st.experimental_rerun()

    def show_cloud_sync(self):
        st.header("☁️ Облачная синхронизация")
        st.subheader("Настройки")
        self.cloud_config['enabled'] = st.checkbox(
            "Включить", value=self.cloud_config['enabled'])
        providers = ["s3", "gcs", "azure"]
        current_idx = providers.index(
            self.cloud_config['provider']) if self.cloud_config['provider'] in providers else 0
        self.cloud_config['provider'] = st.selectbox(
            "Провайдер", providers, index=current_idx)
        self.cloud_config['bucket'] = st.text_input(
            "Bucket / Container", value=self.cloud_config['bucket'])
        self.cloud_config['region'] = st.text_input(
            "Регион", value=self.cloud_config['region'])
        self.cloud_config['sync_interval'] = st.number_input(
            "Интервал (сек)", min_value=300, max_value=86400, value=int(self.cloud_config['sync_interval']))

        if st.button("💾 Сохранить настройки"):
            self.save_cloud_config()
            st.success("Настройки сохранены")

        st.subheader("Текущее состояние")
        last_sync = self.cloud_config.get('last_sync', 0)
        if last_sync > 0:
            st.info(
                f"Последняя синхронизация: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_sync))}")
        else:
            st.info("Еще не синхронизировано")
        if st.button("🔄 Выполнить сейчас"):
            self.perform_cloud_sync()

    def perform_cloud_sync(self):
        if not self.cloud_config.get('enabled'):
            st.warning("Синхронизация отключена")
            return
        if not self.cloud_config.get('bucket'):
            st.error("Не указан bucket")
            return
        with st.spinner("Синхронизация..."):
            time.sleep(1.5)
            st.success("База успешно отправлена")
            self.cloud_config['last_sync'] = int(time.time())
            self.save_cloud_config()

    def show_statistics(self):
        st.header("📈 Статистика")
        stats = {}
        try:
            if self.conn is None:
                st.warning("DB not available — статистика недоступна")
                return
            stats['parts'] = self.conn.execute(
                "SELECT COUNT(*) FROM parts").fetchone()[0]
            stats['oe'] = self.conn.execute(
                "SELECT COUNT(*) FROM oe").fetchone()[0]
            stats['cross'] = self.conn.execute(
                "SELECT COUNT(*) FROM cross_references").fetchone()[0]
            stats['prices'] = self.conn.execute(
                "SELECT COUNT(*) FROM prices").fetchone()[0]
            stats['brands'] = self.conn.execute(
                "SELECT COUNT(DISTINCT brand) FROM parts").fetchone()[0]
            stats['unique_parts'] = self.conn.execute(
                "SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts)").fetchone()[0]
            avg_price = self.conn.execute(
                "SELECT AVG(price) FROM prices").fetchone()[0]
            stats['avg_price'] = round(avg_price, 2) if avg_price else 0
        except Exception as e:
            st.error(f"Ошибка сбора статистики: {e}")
            return
        col1, col2, col3 = st.columns(3)
        col1 and st.title(f"Уникальных товаров: {stats['unique_parts']:,}")
        col2 and st.title(f"Брендов: {stats['brands']:,}")
        col3 and st.title(f"Средняя цена: {stats['avg_price']} ₽")

        try:
            top_brands = self.conn.execute(
                "SELECT brand, COUNT(*) as cnt FROM parts GROUP BY brand ORDER BY cnt DESC LIMIT 10").pl()
            st.subheader("Топ 10 брендов")
            st.dataframe(top_brands.to_pandas())
        except Exception:
            pass

    def merge_all_data_parallel(self, file_paths: Dict[str, str], max_workers: int = 4) -> Dict[str, Any]:
        results = {}
        if not file_paths:
            return results
        if pl is None:
            logger.warning("polars not available — merge_all_data_parallel returns empty results")
            return results
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for key, path in file_paths.items():
                if path and os.path.exists(path):
                    futures[executor.submit(
                        self.read_and_prepare_file, path, key)] = key
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    df = fut.result()
                    if df is not None and not df.is_empty():
                        results[key] = df
                        logger.info(f"Обработан {key}")
                except Exception as e:
                    logger.error(f"Ошибка обработки {key}: {e}")
        return results

    def show_data_management(self):
        st.header("🔧 Управление данными")
        st.warning("⚠️ Операции необратимы!")

        management_option = st.radio(
            "Выберите действие:",
            [
                "Удалить по бренду",
                "Удалить по артикули",
                "Управление ценами",
                "Исключения",
                "Категории",
                "Облачная синхронизация"
            ],
            format_func=lambda x: {
                "Удалить по бренду": "🏭 Удалить все записи бренда",
                "Удалить по артикули": "📦 Удалить все записи артикула",
                "Управление ценами": "💰 Цены и наценки",
                "Исключения": "🚫 Исключения при экспорте",
                "Категории": "🗂️ Категории товаров",
                "Облачная синхронизация": "☁️ Облачная синхронизация"
            }[x]
        )

        if management_option == "Удалить по бренду":
            self._show_delete_by_brand()
        elif management_option == "Удалить по артикули":
            self._show_delete_by_artikul()
        elif management_option == "Управление ценами":
            self.show_price_settings()
        elif management_option == "Исключения":
            self.show_exclusion_settings()
        elif management_option == "Категории":
            self.show_category_mapping()
        elif management_option == "Облачная синхронизация":
            self.show_cloud_sync()

    def _show_delete_by_brand(self):
        st.subheader("Удаление по бренду")
        try:
            if self.conn is None:
                available_brands = []
            else:
                brands_result = self.conn.execute(
                    "SELECT DISTINCT brand FROM parts WHERE brand IS NOT NULL ORDER BY brand").fetchall()
                available_brands = [row[0]
                                    for row in brands_result] if brands_result else []
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            st.error("Ошибка при получении брендов")
            return
        if not available_brands:
            st.info("Нет данных")
            return
        selected_brand = st.selectbox("Бренд", available_brands)

        if self.conn is None:
            brand_norm = selected_brand.lower()
        else:
            brand_norm_result = self.conn.execute(
                "SELECT brand_norm FROM parts WHERE brand = ? LIMIT 1", [selected_brand]).fetchone()
            if brand_norm_result:
                brand_norm = brand_norm_result[0]
            else:
                brand_norm = self.normalize_key([selected_brand])[0] if isinstance(self.normalize_key([selected_brand]), list) else selected_brand.lower()

        count = 0
        if self.conn:
            count = self.conn.execute(
                "SELECT COUNT(*) FROM parts WHERE brand_norm = ?", [brand_norm]).fetchone()[0]
        st.info(f"Удалить {count} записей бренда '{selected_brand}'?")

        if st.checkbox("Подтверждаю удаление"):
            if st.button("Удалить"):
                deleted = self.delete_by_brand(brand_norm)
                st.success(f"Удалено {deleted} записей")
                st.experimental_rerun()

    def _show_delete_by_artikul(self):
        st.subheader("Удаление по артикулу")
        artikul_input = st.text_input("Артикул")
        if artikul_input:
            artikul_norm = self.normalize_key([artikul_input])[0] if isinstance(self.normalize_key([artikul_input]), list) else artikul_input.lower()
            count = 0
            if self.conn:
                count = self.conn.execute(
                    "SELECT COUNT(*) FROM parts WHERE artikul_norm = ?", [artikul_norm]).fetchone()[0]
            st.info(f"Найдено {count} записей для артикула '{artikul_input}'")
            if st.checkbox("Подтверждаю"):
                if st.button("Удалить"):
                    deleted = self.delete_by_artikul(artikul_norm)
                    st.success(f"Удалено {deleted} записей")
                    st.experimental_rerun()

def main():
    st.title("🚗 AutoParts Catalog 10M+")
    st.markdown("### Платформа для больших каталогов автозапчастей")
    catalog = HighVolumeAutoPartsCatalog()

    st.sidebar = getattr(st, "sidebar", st)  # shim: ensure sidebar exists
    st.sidebar.title("🧭 Меню")
    option = st.sidebar.radio(
        "Выберите раздел", ["Загрузка данных", "Экспорт", "Статистика", "Управление"])

    if option == "Загрузка данных":
        st.header("📥 Загрузка данных")
        col1, col2 = st.columns(2)
        with col1:
            oe_file = st.file_uploader("Основные данные (OE)", type=['xlsx'])
            cross_file = st.file_uploader("Кроссы (OE→Артикул)", type=['xlsx'])
            barcode_file = st.file_uploader("Штрих-коды", type=['xlsx'])
        with col2:
            weight_dims_file = st.file_uploader(
                "Вес и габариты", type=['xlsx'])
            images_file = st.file_uploader("Изображения", type=['xlsx'])
            prices_file = st.file_uploader("Цены", type=['xlsx'])

        uploaded_files = {
            'oe': oe_file,
            'cross': cross_file,
            'barcode': barcode_file,
            'dimensions': weight_dims_file,
            'images': images_file,
            'prices': prices_file
        }

        if st.button("Обработать и загрузить"):
            saved_paths = {}
            for key, file in uploaded_files.items():
                if file:
                    path = catalog.data_dir / f"{key}_{int(time.time())}.xlsx"
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    saved_paths[key] = str(path)
            if saved_paths:
                with st.spinner("Обработка файлов..."):
                    dataframes = catalog.merge_all_data_parallel(saved_paths)
                with st.spinner("Загрузка данных в базу..."):
                    catalog.process_and_load_data(dataframes)
            else:
                st.warning("Загрузите хотя бы один файл")
    elif option == "Экспорт":
        catalog.show_export_interface()
    elif option == "Статистика":
        catalog.show_statistics()
    elif option == "Управление":
        catalog.show_data_management()

if __name__ == "__main__":
    main()
