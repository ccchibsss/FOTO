# ==============================================================================
# 🚗 AutoParts Catalog — Умная система управления каталогом автозапчастей
# 
# 🔧 Версия: 1.3 (исправленная + расширяемая)
# 📅 Дата: 2025
# 🎯 Особенности:
#    - Автоматическое добавление новых столбцов
#    - Устранение дубликатов колонок
#    - Безопасный UPSERT в DuckDB
#    - Поддержка больших данных (10M+)
# ==============================================================================

import platform
import sys
import polars as pl
import duckdb
import streamlit as st
import os
import time
import logging
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import json

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ограничение на строки в Excel
EXCEL_ROW_LIMIT = 1_048_576


class HighVolumeAutoPartsCatalog:
    """
    🏭 Ядро системы — управление данными, схемой, бизнес-логикой
    """

    def __init__(self):
        """Инициализация: папки, БД, конфиги"""
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)

        self.cloud_config = self.load_cloud_config()
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(database=str(self.db_path))
        
        # Создаём таблицы и авто-обновляем схему
        self.setup_database()

        # Загрузка правил
        self.price_rules = self.load_price_rules()
        self.exclusion_rules = self.load_exclusion_rules()
        self.category_mapping = self.load_category_mapping()

        # UI
        st.set_page_config(
            page_title="🚗 AutoParts Catalog",
            layout="wide",
            page_icon="🚗",
            initial_sidebar_state="expanded"
        )

    # === 🛠️ КОНФИГУРАЦИЯ ===

    def load_cloud_config(self) -> Dict[str, Any]:
        config_path = self.data_dir / "cloud_config.json"
        default = {"enabled": False, "provider": "s3", "bucket": "", "region": "", "sync_interval": 3600, "last_sync": 0}
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding='utf-8'))
            except Exception as e:
                logger.error(f"Ошибка: {e}")
                return default
        config_path.write_text(json.dumps(default, indent=2, ensure_ascii=False), encoding='utf-8')
        return default

    def save_cloud_config(self):
        config_path = self.data_dir / "cloud_config.json"
        self.cloud_config["last_sync"] = int(time.time())
        config_path.write_text(
            json.dumps(self.cloud_config, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

    def load_price_rules(self) -> Dict[str, Any]:
        path = self.data_dir / "price_rules.json"
        default = {"global_markup": 0.2, "brand_markups": {}, "min_price": 0.0, "max_price": 99999.0}
        if path.exists():
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except Exception as e:
                logger.error(f"Ошибка: {e}")
                return default
        path.write_text(json.dumps(default, indent=2, ensure_ascii=False), encoding='utf-8')
        return default

    def save_price_rules(self):
        path = self.data_dir / "price_rules.json"
        path.write_text(json.dumps(self.price_rules, indent=2, ensure_ascii=False), encoding='utf-8')

    def load_exclusion_rules(self) -> List[str]:
        path = self.data_dir / "exclusion_rules.txt"
        if path.exists():
            try:
                return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            except Exception as e:
                logger.error(f"Ошибка: {e}")
                return []
        path.write_text("Кузов\nСтекла\nМасла", encoding='utf-8')
        return ["Кузов", "Стекла", "Масла"]

    def save_exclusion_rules(self):
        path = self.data_dir / "exclusion_rules.txt"
        path.write_text("\n".join(self.exclusion_rules), encoding='utf-8')

    def load_category_mapping(self) -> Dict[str, str]:
        path = self.data_dir / "category_mapping.txt"
        default = {"Радиатор": "Охлаждение", "Шаровая опора": "Подвеска"}
        if path.exists():
            try:
                mapping = {}
                for line in path.read_text(encoding='utf-8').splitlines():
                    if "|" in line:
                        k, v = line.split("|", 1)
                        mapping[k.strip()] = v.strip()
                return mapping
            except Exception as e:
                logger.error(f"Ошибка: {e}")
                return default
        content = "\n".join(f"{k}|{v}" for k, v in default.items())
        path.write_text(content, encoding='utf-8')
        return default

    def save_category_mapping(self):
        path = self.data_dir / "category_mapping.txt"
        content = "\n".join(f"{k}|{v}" for k, v in self.category_mapping.items())
        path.write_text(content, encoding='utf-8')

    # === 🗃️ РАБОТА С БАЗОЙ ===

    def setup_database(self):
        """Создание таблиц + авто-добавление новых колонок"""
        self._create_oe_data()
        self._create_cross_references()
        self._create_prices()
        self._create_parts_data_with_dynamic_schema()

    def _create_oe_data(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS oe_data (
                oe_number_norm VARCHAR PRIMARY KEY,
                oe_number VARCHAR,
                name VARCHAR,
                applicability VARCHAR,
                category VARCHAR
            )
        """)

    def _create_cross_references(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_references (
                oe_number_norm VARCHAR,
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                PRIMARY KEY (oe_number_norm, artikul_norm, brand_norm)
            )
        """)

    def _create_prices(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                artikul_norm VARCHAR,
                brand_norm VARCHAR,
                price DOUBLE,
                currency VARCHAR DEFAULT 'RUB',
                PRIMARY KEY (artikul_norm, brand_norm)
            )
        """)

    def _create_parts_data_with_dynamic_schema(self):
        """Создание таблицы parts_data с возможностью добавления столбцов"""
        base_sql = """
            CREATE TABLE IF NOT EXISTS parts_data (
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
                dimensions_str VARCHAR,
                image_url VARCHAR,
                description VARCHAR
            )
        """
        self.conn.execute(base_sql)
        self.create_indexes()

    def add_missing_columns(self, df: pl.DataFrame, table_name: str):
        """Добавляет недостающие колонки в таблицу"""
        existing_cols = {r[0]: r[1] for r in self.conn.execute(f"DESCRIBE {table_name}").fetchall()}
        for col in df.columns:
            if col not in existing_cols:
                dtype = df[col].dtype
                duckdb_type = "VARCHAR"
                if dtype in [pl.Int32, pl.Int64]: duckdb_type = "BIGINT"
                elif dtype in [pl.Float32, pl.Float64]: duckdb_type = "DOUBLE"
                elif dtype == pl.Boolean: duckdb_type = "BOOLEAN"
                try:
                    self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {duckdb_type}")
                    logger.info(f"✅ Добавлена колонка: {col} ({duckdb_type}) в {table_name}")
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"⚠️ Не удалось добавить колонку {col}: {e}")

    def create_indexes(self):
        """Создание индексов"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_prices_keys ON prices(artikul_norm, brand_norm)"
        ]
        for idx in indexes:
            try:
                self.conn.execute(idx)
            except Exception as e:
                logger.debug(f"Индекс уже существует или ошибка: {e}")

    # === 🔎 ОБРАБОТКА ДАННЫХ ===

    @staticmethod
    def normalize_key(s: pl.Series) -> pl.Series:
        return (s.fill_null("").cast(pl.Utf8)
                .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\\-\\s]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
                .str.to_lowercase())

    @staticmethod
    def clean_values(s: pl.Series) -> pl.Series:
        return (s.fill_null("").cast(pl.Utf8)
                .str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\\-\\s]", "")
                .str.strip_chars())

    def detect_columns(self, actual_columns: List[str], expected_columns: List[str]) -> Dict[str, str]:
        """Авто-сопоставление колонок"""
        variants = {
            'oe_number': ['oe', 'оe', 'oe номер'],
            'artikul': ['артикул', 'article', 'sku'],
            'brand': ['бренд', 'brand'],
            'name': ['наименование', 'название', 'name'],
            'applicability': ['применимость', 'vehicle'],
            'barcode': ['штрих-код', 'barcode'],
            'multiplicity': ['кратность', 'multiplicity'],
            'length': ['длина', 'length'],
            'width': ['ширина', 'width'],
            'height': ['высота', 'height'],
            'weight': ['вес', 'weight'],
            'image_url': ['ссылка', 'url', 'image'],
            'dimensions_str': ['весогабариты', 'dimensions'],
            'price': ['цена', 'price'],
            'currency': ['валюта', 'currency']
        }
        actual_lower = {col.lower(): col for col in actual_columns}
        mapping = {}
        for expected in expected_columns:
            for variant in variants.get(expected, [expected]):
                for key, orig in actual_lower.items():
                    if variant.lower() in key and orig not in mapping:
                        mapping[orig] = expected
                        break
        return mapping

    def read_and_prepare_file(self, file_path: str, file_type: str) -> pl.DataFrame:
        """Чтение + удаление дубликатов колонок + нормализация"""
        logger.info(f"📄 Обработка: {file_type}")
        try:
            df = pl.read_excel(file_path, engine="calamine")
            if df.is_empty():
                return df

            # === Удаление дубликатов колонок ===
            if len(df.columns) != len(set(df.columns)):
                seen = set()
                new_names = []
                for col in df.columns:
                    new_col = col
                    i = 1
                    while new_col in seen:
                        new_col = f"{col}_{i}"
                        i += 1
                    seen.add(new_col)
                    new_names.append(new_col)
                df = df.rename(dict(zip(df.columns, new_names)))
                logger.info(f"🔄 Исправлены дубли колонок: {df.columns}")

        except Exception as e:
            logger.error(f"❌ Ошибка чтения {file_path}: {e}")
            return pl.DataFrame()

        # Схема ожидаемых колонок
        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'cross': ['oe_number', 'artikul', 'brand'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'prices': ['artikul', 'brand', 'price', 'currency']
        }
        expected = schemas.get(file_type, [])
        mapping = self.detect_columns(df.columns, expected)
        df = df.rename(mapping)

        # Нормализация
        for col in ['artikul', 'brand', 'oe_number']:
            if col in df.columns:
                df = df.with_columns(self.normalize_key(pl.col(col)).alias(f"{col}_norm"))

        return df.unique()

    # === 📥 ЗАГРУЗКА ДАННЫХ ===

    def upsert_data(self, table_name: str, df: pl.DataFrame, pk: List[str]):
        """UPSERT с авто-добавлением колонок"""
        if df.is_empty():
            return

        # Добавляем новые колонки, если нужно
        self.add_missing_columns(df, table_name)

        # Фильтруем только существующие колонки
        table_cols = [r[0] for r in self.conn.execute(f"DESCRIBE {table_name}").fetchall()]
        df = df.select([col for col in df.columns if col in table_cols])

        df = df.unique(subset=pk, keep="first")
        temp_name = f"temp_{int(time.time())}"
        self.conn.register(temp_name, df.to_arrow())

        cols = df.columns
        cols_str = ", ".join(f'"{c}"' for c in cols)
        pk_str = ", ".join(f'"{c}"' for c in pk)
        update_cols = [c for c in cols if c not in pk]

        if update_cols:
            update_clause = ", ".join([f'"{c}" = excluded."{c}"' for c in update_cols])
            action = f"DO UPDATE SET {update_clause}"
        else:
            action = "DO NOTHING"

        sql = f"""
            INSERT INTO {table_name} ({cols_str})
            SELECT {cols_str} FROM {temp_name}
            ON CONFLICT ({pk_str}) {action};
        """

        try:
            self.conn.execute(sql)
            logger.info(f"✅ UPSERT в {table_name}: {len(df)} записей")
        except Exception as e:
            logger.error(f"❌ Ошибка при UPSERT в {table_name}: {e}")
            st.error(f"Ошибка при загрузке в {table_name}")
        finally:
            self.conn.unregister(temp_name)

    def upsert_prices(self, price_df: pl.DataFrame):
        if price_df.is_empty():
            return
        price_df = price_df.with_columns([
            self.normalize_key(pl.col('artikul')).alias('artikul_norm'),
            self.normalize_key(pl.col('brand')).alias('brand_norm')
        ])
        if 'currency' not in price_df.columns:
            price_df = price_df.with_columns(pl.lit('RUB').alias('currency'))
        price_df = price_df.filter(
            (pl.col('price') >= self.price_rules['min_price']) &
            (pl.col('price') <= self.price_rules['max_price'])
        )
        self.upsert_data('prices', price_df, ['artikul_norm', 'brand_norm'])

    def process_and_load_data(self, dataframes: Dict[str, pl.DataFrame]):
        """Основной метод загрузки"""
        st.info("🔄 Загрузка данных...")

        # OE + кроссы
        if 'oe' in dataframes:
            df_oe = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_data = df_oe.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique()
            self.upsert_data('oe_data', oe_data, ['oe_number_norm'])
            cross = df_oe.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        if 'cross' in dataframes:
            df_cross = dataframes['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            cross_data = df_cross.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_data, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Цены
        if 'prices' in dataframes:
            self.upsert_prices(dataframes['prices'])

        # Все остальные данные — в parts_data
        part_updates = []
        for ft in ['barcode', 'dimensions', 'images']:
            if ft in dataframes and not dataframes[ft].is_empty():
                df = dataframes[ft]
                if 'artikul_norm' in df.columns and 'brand_norm' in df.columns:
                    part_updates.append(df)

        if part_updates:
            final_df = pl.concat(part_updates).unique(subset=['artikul_norm', 'brand_norm'], keep='first')
            self.upsert_data('parts_data', final_df, ['artikul_norm', 'brand_norm'])

        st.success("✅ Данные загружены")

    # === 📤 ЭКСПОРТ ===

    def build_export_query(self) -> str:
        return """
        WITH PartDetails AS (
            SELECT cr.artikul_norm, cr.brand_norm,
                   STRING_AGG(DISTINCT o.oe_number, ', ') AS oe_list,
                   ANY_VALUE(o.name) AS name,
                   ANY_VALUE(o.applicability) AS applicability
            FROM cross_references cr
            LEFT JOIN oe_data o ON cr.oe_number_norm = o.oe_number_norm
            GROUP BY cr.artikul_norm, cr.brand_norm
        ),
        AllAnalogs AS (
            SELECT cr1.artikul_norm, cr1.brand_norm,
                   STRING_AGG(DISTINCT p2.artikul, ', ') AS analog_list
            FROM cross_references cr1
            JOIN cross_references cr2 ON cr1.oe_number_norm = cr2.oe_number_norm
            JOIN parts_data p2 ON cr2.artikul_norm = p2.artikul_norm AND cr2.brand_norm = p2.brand_norm
            WHERE NOT (cr1.artikul_norm = p2.artikul_norm AND cr1.brand_norm = p2.brand_norm)
            GROUP BY cr1.artikul_norm, cr1.brand_norm
        ),
        AnalogProps AS (
            SELECT cr2.artikul_norm, cr2.brand_norm,
                   MAX(p.length) AS length, MAX(p.width) AS width, MAX(p.height) AS height,
                   MAX(p.weight) AS weight, ANY_VALUE(p.dimensions_str) AS dimensions_str,
                   ANY_VALUE(pd.name) AS name, ANY_VALUE(pd.applicability) AS applicability
            FROM cross_references cr1
            JOIN cross_references cr2 ON cr1.oe_number_norm = cr2.oe_number_norm
            JOIN parts_data p ON cr2.artikul_norm = p.artikul_norm AND cr2.brand_norm = p.brand_norm
            LEFT JOIN PartDetails pd ON cr2.artikul_norm = pd.artikul_norm AND cr2.brand_norm = pd.brand_norm
            GROUP BY cr2.artikul_norm, cr2.brand_norm
        )
        SELECT
            p.artikul AS "Артикул бренда",
            p.brand AS "Бренд",
            COALESCE(pd.name, pa.name) AS "Наименование",
            COALESCE(pd.applicability, pa.applicability) AS "Применимость",
            p.multiplicity AS "Кратность",
            p.barcode AS "Штрих-код",
            COALESCE(p.length, pa.length) AS "Длина",
            COALESCE(p.width, pa.width) AS "Ширина",
            COALESCE(p.height, pa.height) AS "Высота",
            COALESCE(p.weight, pa.weight) AS "Вес",
            COALESCE(NULLIF(p.dimensions_str, ''), pa.dimensions_str) AS "Весогабариты",
            p.image_url AS "Ссылка на изображение",
            pd.oe_list AS "OE номер",
            aa.analog_list AS "аналоги"
        FROM parts_data p
        LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
        LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
        LEFT JOIN AnalogProps pa ON p.artikul_norm = pa.artikul_norm AND p.brand_norm = pa.brand_norm
        ORDER BY p.brand, p.artikul
        """

    def export_to_csv_optimized(self, output_path: str):
        query = self.build_export_query()
        df = self.conn.execute(query).pl()
        df.write_csv(output_path, separator=";")
        st.success(f"✅ Экспорт в CSV: {output_path}")

    def show_export_interface(self):
        st.header("📤 Экспорт")
        total = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]
        st.info(f"📦 {total:,} артикулов")
        if st.button("Экспорт в CSV"):
            path = self.data_dir / "export.csv"
            self.export_to_csv_optimized(str(path))
            with open(path, "rb") as f:
                st.download_button("⬇️ Скачать", f, "export.csv", "text/csv")

    # === 🎨 UI ===

    def show_price_settings(self):
        st.header("💰 Наценки")
        markup = st.number_input("Общая наценка (%)", 0.0, 100.0, self.price_rules['global_markup'] * 100)
        self.price_rules['global_markup'] = markup / 100
        if st.button("Сохранить"):
            self.save_price_rules()
            st.success("✅ Сохранено")

    def show_data_management(self):
        st.header("🔧 Управление")
        opt = st.radio("Действие", ["Цены", "Исключения", "Категории"])
        if opt == "Цены":
            self.show_price_settings()

    # === 🧩 ВСПОМОГАТЕЛЬНЫЕ ===

    def merge_all_data_parallel(self, file_paths: Dict[str, str]) -> Dict[str, pl.DataFrame]:
        results = {}
        with ThreadPoolExecutor() as ex:
            futures = {ex.submit(self.read_and_prepare_file, fp, ft): ft for ft, fp in file_paths.items()}
            for fut in as_completed(futures):
                ft = futures[fut]
                try:
                    df = fut.result()
                    if not df.is_empty():
                        results[ft] = df
                        logger.info(f"✅ Обработан: {ft}")
                except Exception as e:
                    logger.error(f"❌ Ошибка при {ft}: {e}")
        return results


def main():
    st.title("🚗 AutoParts Catalog")
    catalog = HighVolumeAutoPartsCatalog()

    menu = st.sidebar.radio("Меню", ["Загрузка", "Экспорт", "Управление"])

    if menu == "Загрузка":
        st.header("📥 Загрузка данных")
        files = {}
        cols = st.columns(2)
        with cols[0]:
            files['oe'] = st.file_uploader("1. OE", type=["xlsx"])
            files['cross'] = st.file_uploader("2. Кроссы", type=["xlsx"])
            files['barcode'] = st.file_uploader("3. Штрих-коды", type=["xlsx"])
        with cols[1]:
            files['dimensions'] = st.file_uploader("4. Весогабариты", type=["xlsx"])
            files['images'] = st.file_uploader("5. Изображения", type=["xlsx"])
            files['prices'] = st.file_uploader("6. Цены", type=["xlsx"])

        paths = {}
        for t, f in files.items():
            if f:
                p = catalog.data_dir / f"upload_{t}_{int(time.time())}.xlsx"
                with open(p, "wb") as fb:
                    fb.write(f.getbuffer())
                paths[t] = str(p)

        if st.button("🚀 Загрузить"):
            if not paths:
                st.warning("📎 Ничего не загружено")
            else:
                with st.spinner("Чтение..."):
                    dfs = catalog.merge_all_data_parallel(paths)
                if dfs:
                    with st.spinner("Загрузка в базу..."):
                        catalog.process_and_load_data(dfs)
                else:
                    st.error("❌ Ошибка")

    elif menu == "Экспорт":
        catalog.show_export_interface()
    elif menu == "Управление":
        catalog.show_data_management()


if __name__ == "__main__":
    main()
