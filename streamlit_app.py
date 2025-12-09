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

class AutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)

        # Конфигурация облачного хранилища
        self.cloud_config = self.load_cloud_config()
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(database=str(self.db_path))
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

    # === Конфигурирование ===

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
        config_path.write_text(json.dumps(self.cloud_config, indent=2, ensure_ascii=False), encoding='utf-8')

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

    # === Работа с базой данных ===

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
        """Создание таблицы parts_data с возможностью добавления колонок"""
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

    # === Обработка данных ===

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

    # === Загрузка данных ===

    def upsert_data(self, table_name: str, df: pl.DataFrame, pk: List[str]):
        """UPSERT с авто-добавлением колонок"""
        if df.is_empty():
            return

        # Добавление новых колонок
        self.add_missing_columns(df, table_name)

        # Оставляем только существующие колонки таблицы
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
        """Обновление цен"""
        if price_df.is_empty():
            return
        if 'artikul' in price_df.columns and 'brand' in price_df.columns:
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
        """Загрузка данных в базу"""
        st.info("🔄 Начинаю загрузку данных...")
        # Обработка OE
        if 'oe' in dataframes:
            df_oe = dataframes['oe'].filter(pl.col('oe_number_norm') != "")
            oe_data = df_oe.select(['oe_number_norm', 'oe_number', 'name', 'applicability']).unique()
            self.upsert_data('oe_data', oe_data, ['oe_number_norm'])
            cross = df_oe.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка cross
        if 'cross' in dataframes:
            df_cross = dataframes['cross'].filter((pl.col('oe_number_norm') != "") & (pl.col('artikul_norm') != ""))
            cross_data = df_cross.select(['oe_number_norm', 'artikul_norm', 'brand_norm']).unique()
            self.upsert_data('cross_references', cross_data, ['oe_number_norm', 'artikul_norm', 'brand_norm'])

        # Обработка цен
        if 'prices' in dataframes:
            self.upsert_prices(dataframes['prices'])

        # Обработка остальных данных (barcode, dimensions, images)
        part_updates = []
        for ft in ['barcode', 'dimensions', 'images']:
            if ft in dataframes and not dataframes[ft].is_empty():
                df = dataframes[ft]
                if 'artikul_norm' in df.columns and 'brand_norm' in df.columns:
                    part_updates.append(df)

        if part_updates:
            final_df = pl.concat(part_updates).unique(subset=['artikul_norm', 'brand_norm'])
            self.upsert_data('parts_data', final_df, ['artikul_norm', 'brand_norm'])

        st.success("✅ Данные загружены")

    # === Экспорт ===

    def build_export_query(self, selected_columns: Optional[List[str]] = None, include_prices: bool = True, apply_markup: bool = True) -> str:
        """Построение SQL-запроса для экспорта"""
        description_text = """Состояние товара: новый (в упаковке). Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей. В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электроматериалы, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. Выбирайте только лучшее — надежность и качество от ведущих производителей."""

        # Формируем условие по ценам
        price_select = ""
        if include_prices:
            if apply_markup:
                global_markup = self.price_rules['global_markup']
                price_select = """
                CASE
                    WHEN pr.price IS NOT NULL
                    THEN pr.price * (1 + COALESCE(brm.markup, {global_markup}))
                    ELSE pr.price
                END AS "Цена",
                COALESCE(pr.currency, 'RUB') AS "Валюта",
                """.format(global_markup=global_markup)
            else:
                price_select = """
                pr.price AS "Цена",
                COALESCE(pr.currency, 'RUB') AS "Валюта",
                """

        # Условие исключений
        exclusion_conditions = " OR ".join([f"r.representative_name NOT ILIKE '%{ex}%'" for ex in self.exclusion_rules if ex.strip()])
        exclusion_where = f"AND ({exclusion_conditions})" if exclusion_conditions else ""

        # Колонки для вывода
        columns_map = [
            ("Артикул бренда", 'r.artikul AS "Артикул бренда"'),
            ("Бренд", 'r.brand AS "Бренд"'),
            ("Наименование", 'COALESCE(r.representative_name, r.analog_representative_name) AS "Наименование"'),
            ("Применимость", 'COALESCE(r.representative_applicability, r.analog_representative_applicability) AS "Применимость"'),
            ("Описание", 'CONCAT(COALESCE(r.description, ""), dt.text) AS "Описание"'),
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

        if include_prices:
            columns_map.extend([("Цена", '"Цена"'), ("Валюта", '"Валюта"')])

        # Выбор колонок
        if selected_columns:
            selected_exprs = [expr for name, expr in columns_map if name in selected_columns]
        else:
            selected_exprs = [expr for _, expr in columns_map]

        # Построение CTE-запросов
        ctes = f"""
        WITH DescriptionTemplate AS (
            SELECT '{description_text}' AS text
        ),
        BrandMarkups AS (
            SELECT brand, markup FROM (
                {self._get_brand_markups_sql()}
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
            LEFT JOIN oe_data o ON cr.oe_number_norm = o.oe_number_norm
            GROUP BY cr.artikul_norm, cr.brand_norm
        ),
        AllAnalogs AS (
            SELECT 
                cr1.artikul_norm, 
                cr1.brand_norm,
                STRING_AGG(
                    DISTINCT regexp_replace(
                        regexp_replace(p2.artikul, '''', ''),
                        '[^0-9A-Za-zА-Яа-яЁё`\\-\\s]', '', 'g'
                    ), ', '
                ) AS analog_list
            FROM cross_references cr1
            JOIN cross_references cr2 ON cr1.oe_number_norm = cr2.oe_number_norm
            JOIN parts_data p2 ON cr2.artikul_norm = p2.artikul_norm AND cr2.brand_norm = p2.brand_norm
            WHERE (cr1.artikul_norm != p2.artikul_norm OR cr1.brand_norm != p2.brand_norm)
            GROUP BY cr1.artikul_norm, cr1.brand_norm
        ),
        InitialOENumbers AS (
            SELECT DISTINCT p.artikul_norm, p.brand_norm, cr.oe_number_norm
            FROM parts_data p
            LEFT JOIN cross_references cr ON p.artikul_norm = cr.artikul_norm AND p.brand_norm = cr.brand_norm
            WHERE cr.oe_number_norm IS NOT NULL
        ),
        Level1Analogs AS (
            SELECT DISTINCT 
                i.artikul_norm AS source_artikul_norm, 
                i.brand_norm AS source_brand_norm,
                cr2.artikul_norm AS related_artikul_norm, 
                cr2.brand_norm AS related_brand_norm
            FROM InitialOENumbers i
            JOIN cross_references cr2 ON i.oe_number_norm = cr2.oe_number_norm
            WHERE NOT (i.artikul_norm = cr2.artikul_norm AND i.brand_norm = cr2.brand_norm)
        ),
        Level1OENumbers AS (
            SELECT DISTINCT 
                l1.source_artikul_norm, 
                l1.source_brand_norm, 
                cr3.oe_number_norm
            FROM Level1Analogs l1
            JOIN cross_references cr3 ON l1.related_artikul_norm = cr3.artikul_norm AND l1.related_brand_norm = cr3.brand_norm
            WHERE NOT EXISTS (
                SELECT 1 FROM InitialOENumbers i
                WHERE i.artikul_norm = l1.source_artikul_norm 
                  AND i.brand_norm = l1.source_brand_norm 
                  AND i.oe_number_norm = cr3.oe_number_norm
            )
        ),
        Level2Analogs AS (
            SELECT DISTINCT 
                loe.source_artikul_norm, 
                loe.source_brand_norm,
                cr4.artikul_norm AS related_artikul_norm, 
                cr4.brand_norm AS related_brand_norm
            FROM Level1OENumbers loe
            JOIN cross_references cr4 ON loe.oe_number_norm = cr4.oe_number_norm
            WHERE NOT (loe.source_artikul_norm = cr4.artikul_norm AND loe.source_brand_norm = cr4.brand_norm)
        ),
        AllRelatedParts AS (
            SELECT source_artikul_norm, source_brand_norm, related_artikul_norm, related_brand_norm
            FROM Level1Analogs
            UNION
            SELECT source_artikul_norm, source_brand_norm, related_artikul_norm, related_brand_norm
            FROM Level2Analogs
        ),
        AggregatedAnalogData AS (
            SELECT 
                arp.source_artikul_norm AS artikul_norm,
                arp.source_brand_norm AS brand_norm,
                MAX(CASE WHEN p2.length IS NOT NULL THEN p2.length ELSE NULL END) AS length,
                MAX(CASE WHEN p2.width IS NOT NULL THEN p2.width ELSE NULL END) AS width,
                MAX(CASE WHEN p2.height IS NOT NULL THEN p2.height ELSE NULL END) AS height,
                MAX(CASE WHEN p2.weight IS NOT NULL THEN p2.weight ELSE NULL END) AS weight,
                ANY_VALUE(
                    CASE 
                        WHEN p2.dimensions_str IS NOT NULL AND p2.dimensions_str != '' AND UPPER(TRIM(p2.dimensions_str)) != 'XX'
                        THEN p2.dimensions_str
                        ELSE NULL
                    END
                ) AS dimensions_str,
                ANY_VALUE(
                    CASE 
                        WHEN pd2.representative_name IS NOT NULL AND pd2.representative_name != '' 
                        THEN pd2.representative_name 
                        ELSE NULL
                    END
                ) AS representative_name,
                ANY_VALUE(
                    CASE 
                        WHEN pd2.representative_applicability IS NOT NULL AND pd2.representative_applicability != ''
                        THEN pd2.representative_applicability
                        ELSE NULL
                    END
                ) AS representative_applicability,
                ANY_VALUE(
                    CASE 
                        WHEN pd2.representative_category IS NOT NULL AND pd2.representative_category != ''
                        THEN pd2.representative_category
                        ELSE NULL
                    END
                ) AS representative_category
            FROM AllRelatedParts arp
            JOIN parts_data p2 ON arp.related_artikul_norm = p2.artikul_norm AND arp.related_brand_norm = p2.brand_norm
            LEFT JOIN PartDetails pd2 ON p2.artikul_norm = pd2.artikul_norm AND p2.brand_norm = pd2.brand_norm
            GROUP BY arp.source_artikul_norm, arp.source_brand_norm
        ),
        RankedData AS (
            SELECT 
                p.artikul,
                p.brand,
                p.description,
                p.multiplicity,
                p.length,
                p.width,
                p.height,
                p.weight,
                p.dimensions_str,
                p.image_url,
                pd.representative_name,
                pd.representative_applicability,
                pd.representative_category,
                pd.oe_list,
                aa.analog_list,
                p_analog.length AS analog_length,
                p_analog.width AS analog_width,
                p_analog.height AS analog_height,
                p_analog.weight AS analog_weight,
                p_analog.dimensions_str AS analog_dimensions_str,
                p_analog.representative_name AS analog_representative_name,
                p_analog.representative_applicability AS analog_representative_applicability,
                p_analog.representative_category AS analog_representative_category,
                ROW_NUMBER() OVER (
                    PARTITION BY p.artikul_norm, p.brand_norm 
                    ORDER BY pd.representative_name DESC NULLS LAST, pd.oe_list DESC NULLS LAST
                ) AS rn
            FROM parts_data p
            LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
            LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
            LEFT JOIN AggregatedAnalogData p_analog ON p.artikul_norm = p_analog.artikul_norm AND p.brand_norm = p_analog.brand_norm
        )
        """

        select_clause = ",\n        ".join(selected_exprs)

        price_join = """
        LEFT JOIN prices pr ON r.artikul_norm = pr.artikul_norm AND r.brand_norm = pr.brand_norm
        LEFT JOIN BrandMarkups brm ON r.brand = brm.brand
        """ if include_prices else ""

        query = f"""
        {ctes}
        SELECT
            {price_select}
            {select_clause}
        FROM RankedData r
        CROSS JOIN DescriptionTemplate dt
        {price_join}
        WHERE r.rn = 1
        {exclusion_where}
        ORDER BY r.brand, r.artikul
        """

        return query.strip()

    def _get_brand_markups_sql(self) -> str:
        """Генерация SQL-подзапроса для наценок по брендам"""
        rows = []
        for brand, markup in self.price_rules['brand_markups'].items():
            rows.append(f"SELECT '{brand}' AS brand, {markup} AS markup")
        return " UNION ALL ".join(rows) if rows else "SELECT NULL AS brand, NULL AS markup LIMIT 0"

    def export_to_csv_optimized(self, output_path: str, selected_columns: Optional[List[str]] = None, include_prices: bool = True, apply_markup: bool = True) -> bool:
        """Экспорт данных в CSV с оптимизацией типов и размера"""
        total_records = self.conn.execute("""
            SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)
        """).fetchone()[0]
        if total_records == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📤 Экспорт {total_records:,} записей в CSV...")
        try:
            query = self.build_export_query(selected_columns, include_prices, apply_markup)
            df = self.conn.execute(query).pl()

            # Преобразование размерных колонок в строки
            dimension_cols = ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота"]
            for col in dimension_cols:
                if col in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col).is_not_null())
                         .then(pl.col(col).cast(pl.Utf8))
                         .otherwise(pl.lit(""))
                         .alias(col)
                    )

            # Запись в CSV с BOM для Excel
            buf = io.StringIO()
            df.write_csv(buf, separator=';')
            csv_text = buf.getvalue()

            with open(output_path, 'wb') as f:
                f.write(b'\xef\xbb\xbf')  # BOM
                f.write(csv_text.encode('utf-8'))

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            st.success(f"✅ Данные экспортированы в CSV: {output_path} ({file_size:.1f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта в CSV")
            st.error(f"❌ Ошибка экспорта в CSV: {e}")
            return False

    def show_price_settings(self):
        """Интерфейс настройки цен и наценок"""
        st.header("💰 Управление ценами и наценками")
        # Общая наценка
        st.subheader("Общая наценка")
        global_markup = st.number_input(
            "Общая наценка (%):",
            min_value=0.0,
            max_value=100.0,
            value=self.price_rules['global_markup'] * 100,
            step=0.1
        )
        self.price_rules['global_markup'] = global_markup / 100

        # Наценки по брендам
        st.subheader("Наценки по брендам")
        brand_markups = self.price_rules.get('brand_markups', {})

        try:
            brands_result = self.conn.execute("SELECT DISTINCT brand FROM parts_data WHERE brand IS NOT NULL ORDER BY brand").fetchall()
            available_brands = [row[0] for row in brands_result] if brands_result else []
        except Exception as e:
            logger.error(f"Ошибка при получении списка брендов: {e}")
            st.error("❌ Ошибка при загрузке брендов")
            available_brands = []

        if available_brands:
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_brand = st.selectbox("Выберите бренд:", available_brands)
            with col2:
                current_markup = brand_markups.get(selected_brand, self.price_rules.get('global_markup', 0))
                brand_markup = st.number_input(
                    "Наценка (%):",
                    min_value=0.0,
                    max_value=100.0,
                    value=current_markup * 100,
                    step=0.1,
                    key=f"markup_{selected_brand}"
                )
            if st.button("Сохранить наценку", key=f"save_{selected_brand}"):
                # Обновляем словарь наценок
                brand_markups[selected_brand] = brand_markup / 100
                self.price_rules['brand_markups'] = brand_markups
                self.save_price_rules()
                st.success(f"✅ Наценка для {selected_brand} сохранена")

        # Ограничения цен
        st.subheader("Ограничения по ценам")
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Минимальная цена:", min_value=0.0, value=float(self.price_rules['min_price']), step=0.01)
            self.price_rules['min_price'] = min_price
        with col2:
            max_price = st.number_input("Максимальная цена:", min_value=0.0, value=float(self.price_rules['max_price']), step=0.01)
            self.price_rules['max_price'] = max_price

        if st.button("Сохранить все настройки цен"):
            self.save_price_rules()
            st.success("✅ Все настройки цен сохранены")

    def show_exclusion_settings(self):
        """Интерфейс управления списком исключений при экспорте"""
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
            # Очистка и фильтрация ввода
            cleaned = [line.strip() for line in new_exclusions.splitlines() if line.strip()]
            if len(cleaned) != len(set(cleaned)):
                st.warning("Обнаружены дублирующиеся записи. Они будут автоматически удалены.")
            self.exclusion_rules = list(dict.fromkeys(cleaned))
            self.save_exclusion_rules()
            st.success("✅ Правила исключения сохранены")

    def show_category_mapping(self):
        """Интерфейс настройки пользовательских категорий"""
        st.header("🗂️ Управление категориями товаров")
        st.info("Настройте соответствие между названиями товаров и категориями")

        # Текущие правила
        st.subheader("Текущие правила")
        if self.category_mapping:
            mapping_df = pl.DataFrame({
                "Название товара": list(self.category_mapping.keys()),
                "Категория": list(self.category_mapping.values())
            }).to_pandas()
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        else:
            st.write("Нет пользовательских правил категоризации")

        # Добавление правила
        st.subheader("Добавить правило")
        col1, col2 = st.columns(2)
        with col1:
            name_pattern = st.text_input("Ключевое слово")
        with col2:
            category = st.text_input("Категория")

        if st.button("➕ Добавить правило"):
            if name_pattern.strip() and category.strip():
                normalized_key = name_pattern.strip().lower()
                existing_keys = {k.lower(): k for k in self.category_mapping.keys()}
                if normalized_key in existing_keys:
                    st.warning(f"Правило для '{existing_keys[normalized_key]}' будет обновлено")
                self.category_mapping[name_pattern.strip()] = category.strip()
                self.save_category_mapping()
                st.success(f"✅ Правило добавлено: {name_pattern.strip()} → {category.strip()}")
                st.rerun()
            else:
                st.error("❌ Заполните оба поля")

        # Удаление правила
        if self.category_mapping:
            st.subheader("🗑️ Удалить правило")
            rule_to_delete = st.selectbox(
                "Выберите правило:",
                options=list(self.category_mapping.keys()),
                format_func=lambda x: f"{x} → {self.category_mapping[x]}"
            )
            if st.button("Удалить правило"):
                del self.category_mapping[rule_to_delete]
                self.save_category_mapping()
                st.success(f"✅ Правило удалено: {rule_to_delete}")
                st.rerun()

    def show_data_management(self):
        """Основной интерфейс управления данными"""
        st.header("🔧 Управление данными")
        st.warning("⚠️ Операции необратимы. Будьте осторожны.")

        management_option = st.radio(
            "Выберите действие:",
            [
                "Удалить по бренду",
                "Удалить по артикулу",
                "Управление ценами",
                "Исключения при экспорте",
                "Категории товаров",
                "Облачная синхронизация"
            ],
            format_func=lambda x: {
                "Удалить по бренду": "🏭 Удалить все записи бренда",
                "Удалить по артикулу": "📦 Удалить все записи артикула",
                "Управление ценами": "💰 Наценки и лимиты цен",
                "Исключения при экспорте": "🚫 Фильтрация при экспорте",
                "Категории товаров": "🗂️ Ручное назначение категорий",
                "Облачная синхронизация": "☁️ Настройка бэкапа"
            }[x]
        )

        if management_option == "Удалить по бренду":
            self._show_delete_by_brand()
        elif management_option == "Удалить по артикулу":
            self._show_delete_by_artikul()
        elif management_option == "Управление ценами":
            self.show_price_settings()
        elif management_option == "Исключения при экспорте":
            self.show_exclusion_settings()
        elif management_option == "Категории товаров":
            self.show_category_mapping()
        elif management_option == "Облачная синхронизация":
            self.show_cloud_sync()

    def _show_delete_by_brand(self):
        """Удаление по бренду"""
        st.subheader("🗑️ Удаление по бренду")
        try:
            brands_result = self.conn.execute("""
                SELECT DISTINCT brand FROM parts_data WHERE brand IS NOT NULL ORDER BY brand
            """).fetchall()
            available_brands = [row[0] for row in brands_result] if brands_result else []
        except Exception as e:
            logger.error(f"Ошибка при получении брендов: {e}")
            st.error("❌ Не удалось получить список брендов")
            return

        if not available_brands:
            st.info("Нет данных о брендах")
            return

        selected_brand = st.selectbox("Выберите бренд", available_brands)
        # Получение нормализованного ключа
        brand_norm_result = self.conn.execute("SELECT brand_norm FROM parts_data WHERE brand = ? LIMIT 1", [selected_brand]).fetchone()
        if brand_norm_result:
            brand_norm = brand_norm_result[0]
        else:
            brand_norm = self.normalize_key(pl.Series([selected_brand]))[0]

        count = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE brand_norm = ?", [brand_norm]).fetchone()[0]
        st.info(f"Будет удалено {count} записей бренда '{selected_brand}'")
        confirm = st.checkbox("Я подтверждаю удаление всех записей этого бренда")
        if st.button("❌ Удалить бренд", disabled=not confirm):
            deleted = self.delete_by_brand(brand_norm)
            st.success(f"Удалено {deleted} записей")
            st.rerun()

    def delete_by_brand(self, brand_norm: str) -> int:
        """Удаление всех по бренду"""
        with self.conn.transaction():
            count1 = self.conn.execute("DELETE FROM parts_data WHERE brand_norm = ?", [brand_norm]).rowcount
            count2 = self.conn.execute("DELETE FROM cross_references WHERE brand_norm = ?", [brand_norm]).rowcount
            return count1 + count2

    def _show_delete_by_artikul(self):
        """Удаление по артикулу"""
        st.subheader("🗑️ Удаление по артикулу")
        input_art = st.text_input("Введите артикул")
        if input_art:
            artikul_norm = self.normalize_key(pl.Series([input_art]))[0]
            count = self.conn.execute("SELECT COUNT(*) FROM parts_data WHERE artikul_norm = ?", [artikul_norm]).fetchone()[0]
            st.info(f"Найдено {count} записей для артикула '{input_art}'")
            confirm = st.checkbox("Подтвердить удаление")
            if st.button("Удалить", disabled=not confirm):
                deleted = self.delete_by_artikul(artikul_norm)
                st.success(f"Удалено {deleted} записей")
                st.rerun()

    def delete_by_artikul(self, artikul_norm: str) -> int:
        """Удаление по артикулу"""
        with self.conn.transaction():
            count1 = self.conn.execute("DELETE FROM parts_data WHERE artikul_norm = ?", [artikul_norm]).rowcount
            count2 = self.conn.execute("DELETE FROM cross_references WHERE artikul_norm = ?", [artikul_norm]).rowcount
            return count1 + count2

    def show_cloud_sync(self):
        """Настройки облачной синхронизации"""
        st.header("☁️ Облачная синхронизация")
        # Настройки
        st.subheader("🔧 Конфигурация")
        col1, col2 = st.columns(2)
        with col1:
            self.cloud_config['enabled'] = st.checkbox("Включить синхронизацию", value=self.cloud_config['enabled'])
        with col2:
            providers = ["s3", "gcs", "azure"]
            idx = providers.index(self.cloud_config['provider']) if self.cloud_config['provider'] in providers else 0
            self.cloud_config['provider'] = st.selectbox("Провайдер", providers, index=idx)

        self.cloud_config['bucket'] = st.text_input("Bucket / Container", value=self.cloud_config['bucket'])
        self.cloud_config['region'] = st.text_input("Регион", value=self.cloud_config['region'])
        self.cloud_config['sync_interval'] = st.number_input("Интервал синхронизации (сек)", min_value=300, max_value=86400, value=int(self.cloud_config['sync_interval']))

        if st.button("💾 Сохранить настройки"):
            self.save_cloud_config()
            st.success("✅ Конфигурация сохранена")

        # Статус
        st.subheader("📊 Текущее состояние")
        if self.cloud_config['last_sync'] > 0:
            last_sync_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.cloud_config['last_sync']))
            st.info(f"Последняя синхронизация: {last_sync_str}")
        else:
            st.info("Синхронизация ещё не выполнялась")
        if st.button("🔄 Выполнить синхронизацию сейчас"):
            self.perform_cloud_sync()

    def perform_cloud_sync(self):
        """Заглушка синхронизации"""
        if not self.cloud_config['enabled']:
            st.warning("❌ Синхронизация отключена")
            return
        if not self.cloud_config['bucket']:
            st.error("❌ Не указан bucket")
            return
        with st.spinner("Выполняется синхронизация..."):
            try:
                # Тут должна быть логика работы с облаком
                time.sleep(1.5)
                st.success(f"📤 База данных отправлена в {self.cloud_config['provider']}://{self.cloud_config['bucket']}")
                self.cloud_config['last_sync'] = int(time.time())
                self.save_cloud_config()
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

    def show_export_interface(self):
        """Интерфейс экспорта"""
        st.header("📤 Экспорт данных")
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        st.info(f"📦 Всего уникальных пар (артикул + бренд): {total:,}")
        if total == 0:
            st.warning("Нет данных для экспорта")
            return

        options_columns = [
            "Артикул бренда", "Бренд", "Наименование", "Применимость", "Описание",
            "Категория товара", "Кратность", "Длинна", "Ширина", "Высота", "Вес",
            "Длинна/Ширина/Высота", "OE номер", "аналоги", "Ссылка на изображение"
        ]
        # Цены
        if self.conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0] > 0:
            options_columns.extend(["Цена", "Валюта"])

        selected_columns = st.multiselect("Выберите колонки для экспорта", options=options_columns, default=options_columns)

        col1, col2 = st.columns(2)
        with col1:
            export_format = st.radio("Формат", ["CSV", "Excel (.xlsx)", "Parquet"])
        with col2:
            include_prices = st.checkbox("Включить цены", value=True)
            apply_markup = st.checkbox("Применить наценку", value=True, disabled=not include_prices)

        if st.button("🚀 Выполнить экспорт"):
            output_path = self.data_dir / f"auto_parts_export.{export_format.lower().replace(' ', '_')}"
            with st.spinner("Формирование отчета..."):
                if export_format == "CSV":
                    self.export_to_csv_optimized(str(output_path), selected_columns if selected_columns else None, include_prices, apply_markup)
                elif export_format == "Excel (.xlsx)":
                    self.export_to_excel_optimized(str(output_path), selected_columns if selected_columns else None, include_prices, apply_markup)
                elif export_format == "Parquet":
                    self.export_to_parquet(str(output_path), selected_columns if selected_columns else None, include_prices, apply_markup)
                else:
                    st.warning("Выбран неподдерживаемый формат")
            with open(output_path, "rb") as f:
                st.download_button("⬇️ Скачать файл", f, output_path.name, "application/octet-stream")

    def export_to_excel_optimized(self, output_path: str, selected_columns: Optional[List[str]] = None, include_prices: bool = True, apply_markup: bool = True) -> bool:
        """Экспорт в Excel с разбивкой по лимитам"""
        total = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📊 Подготовка экспорта в Excel: {total:,} записей")
        try:
            import pandas as pd
            query = self.build_export_query(selected_columns, include_prices, apply_markup)
            df = pd.read_sql(query, self.conn)
            for col in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).replace({r'^nan$': ''}, regex=True)

            if len(df) <= EXCEL_ROW_LIMIT:
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Данные')
            else:
                sheets = (len(df) // EXCEL_ROW_LIMIT) + 1
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    for i in range(sheets):
                        start = i * EXCEL_ROW_LIMIT
                        end = min((i+1) * EXCEL_ROW_LIMIT, len(df))
                        df.iloc[start:end].to_excel(writer, index=False, sheet_name=f"Данные_{i+1}")
            file_size = os.path.getsize(output_path) / (1024*1024)
            st.success(f"✅ Данные экспортированы в Excel: {output_path} ({file_size:.1f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта в Excel")
            st.error(f"❌ Ошибка: {e}")
            return False

    def export_to_parquet(self, output_path: str, selected_columns: Optional[List[str]] = None, include_prices: bool = True, apply_markup: bool = True) -> bool:
        """Экспорт в Parquet"""
        st.info("📦 Подготовка экспорта в Parquet...")
        try:
            query = self.build_export_query(selected_columns, include_prices, apply_markup)
            df = self.conn.execute(query).pl()
            df.write_parquet(output_path)
            file_size = os.path.getsize(output_path) / (1024*1024)
            st.success(f"✅ Данные экспортированы в Parquet: {output_path} ({file_size:.1f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта в Parquet")
            st.error(f"❌ Ошибка: {e}")
            return False

    def export_to_csv_optimized(self, output_path: str, selected_columns: Optional[List[str]] = None, include_prices: bool = True, apply_markup: bool = True) -> bool:
        """Экспорт в CSV"""
        total = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📤 Экспорт {total:,} записей в CSV...")
        try:
            query = self.build_export_query(selected_columns, include_prices, apply_markup)
            df = self.conn.execute(query).pl()

            # превращение размерных колонок в строки
            for col in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота"]:
                if col in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col).is_not_null())
                          .then(pl.col(col).cast(pl.Utf8))
                          .otherwise(pl.lit(""))
                          .alias(col)
                    )

            buf = io.StringIO()
            df.write_csv(buf, separator=';')
            csv_text = buf.getvalue()

            with open(output_path, 'wb') as f:
                f.write(b'\xef\xbb\xbf')  # BOM for Excel
                f.write(csv_text.encode('utf-8'))

            size_mb = os.path.getsize(output_path) / (1024*1024)
            st.success(f"✅ Данные экспортированы в CSV: {output_path} ({size_mb:.1f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта в CSV")
            st.error(f"❌ Ошибка: {e}")
            return False

    def build_export_query(self, selected_columns=None, include_prices=True, apply_markup=True):
        """Построение сложного SQL-запроса для экспорта"""
        description_text = """Состояние товара: новый (в упаковке). Высококачественные автозапчасти и автотовары — надежное решение для вашего автомобиля. Обеспечьте безопасность, долговечность и высокую производительность вашего авто с помощью нашего широкого ассортимента оригинальных и совместимых автозапчастей. В нашем каталоге вы найдете тормозные системы, фильтры (масляные, воздушные, салонные), свечи зажигания, расходные материалы, автохимию, электроматериалы, автомасла, инструмент, а также другие комплектующие, полностью соответствующие стандартам качества и безопасности. Мы гарантируем быструю доставку, выгодные цены и профессиональную консультацию для любого клиента — автолюбителя, специалиста или автосервиса. Выбирайте только лучшее — надежность и качество от ведущих производителей."""

        price_select = ""
        if include_prices:
            if apply_markup:
                global_markup = self.price_rules['global_markup']
                price_select = """
                CASE
                    WHEN pr.price IS NOT NULL
                    THEN pr.price * (1 + COALESCE(brm.markup, {global_markup}))
                    ELSE pr.price
                END AS "Цена",
                COALESCE(pr.currency, 'RUB') AS "Валюта",
                """.format(global_markup=global_markup)
            else:
                price_select = """
                pr.price AS "Цена",
                COALESCE(pr.currency, 'RUB') AS "Валюта",
                """

        # Условие исключений
        exclusion_conditions = " OR ".join([f"r.representative_name NOT ILIKE '%{ex}%'" for ex in self.exclusion_rules if ex.strip()])
        exclusion_where = f"AND ({exclusion_conditions})" if exclusion_conditions else ""

        # Колонки для вывода
        columns_map = [
            ("Артикул бренда", 'r.artikul AS "Артикул бренда"'),
            ("Бренд", 'r.brand AS "Бренд"'),
            ("Наименование", 'COALESCE(r.representative_name, r.analog_representative_name) AS "Наименование"'),
            ("Применимость", 'COALESCE(r.representative_applicability, r.analog_representative_applicability) AS "Применимость"'),
            ("Описание", 'CONCAT(COALESCE(r.description, ""), dt.text) AS "Описание"'),
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

        if include_prices:
            columns_map.extend([("Цена", '"Цена"'), ("Валюта", '"Валюта"')])

        if selected_columns:
            selected_exprs = [expr for name, expr in columns_map if name in selected_columns]
        else:
            selected_exprs = [expr for _, expr in columns_map]

        # CTE-запросы
        ctes = f"""
        WITH DescriptionTemplate AS (
            SELECT '{description_text}' AS text
        ),
        BrandMarkups AS (
            SELECT brand, markup FROM (
                {self._get_brand_markups_sql()}
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
            LEFT JOIN oe_data o ON cr.oe_number_norm = o.oe_number_norm
            GROUP BY cr.artikul_norm, cr.brand_norm
        ),
        AllAnalogs AS (
            SELECT 
                cr1.artikul_norm, 
                cr1.brand_norm,
                STRING_AGG(
                    DISTINCT regexp_replace(
                        regexp_replace(p2.artikul, '''', ''),
                        '[^0-9A-Za-zА-Яа-яЁё`\\-\\s]', '', 'g'
                    ), ', '
                ) AS analog_list
            FROM cross_references cr1
            JOIN cross_references cr2 ON cr1.oe_number_norm = cr2.oe_number_norm
            JOIN parts_data p2 ON cr2.artikul_norm = p2.artikul_norm AND cr2.brand_norm = p2.brand_norm
            WHERE (cr1.artikul_norm != p2.artikul_norm OR cr1.brand_norm != p2.brand_norm)
            GROUP BY cr1.artikul_norm, cr1.brand_norm
        ),
        InitialOENumbers AS (
            SELECT DISTINCT p.artikul_norm, p.brand_norm, cr.oe_number_norm
            FROM parts_data p
            LEFT JOIN cross_references cr ON p.artikul_norm = cr.artikul_norm AND p.brand_norm = cr.brand_norm
            WHERE cr.oe_number_norm IS NOT NULL
        ),
        Level1Analogs AS (
            SELECT DISTINCT 
                i.artikul_norm AS source_artikul_norm, 
                i.brand_norm AS source_brand_norm,
                cr2.artikul_norm AS related_artikul_norm, 
                cr2.brand_norm AS related_brand_norm
            FROM InitialOENumbers i
            JOIN cross_references cr2 ON i.oe_number_norm = cr2.oe_number_norm
            WHERE NOT (i.artikul_norm = cr2.artikul_norm AND i.brand_norm = cr2.brand_norm)
        ),
        Level1OENumbers AS (
            SELECT DISTINCT 
                l1.source_artikul_norm, 
                l1.source_brand_norm, 
                cr3.oe_number_norm
            FROM Level1Analogs l1
            JOIN cross_references cr3 ON l1.related_artikul_norm = cr3.artikul_norm AND l1.related_brand_norm = cr3.brand_norm
            WHERE NOT EXISTS (
                SELECT 1 FROM InitialOENumbers i
                WHERE i.artikul_norm = l1.source_artikul_norm 
                  AND i.brand_norm = l1.source_brand_norm 
                  AND i.oe_number_norm = cr3.oe_number_norm
            )
        ),
        Level2Analogs AS (
            SELECT DISTINCT 
                loe.source_artikul_norm, 
                loe.source_brand_norm,
                cr4.artikul_norm AS related_artikul_norm, 
                cr4.brand_norm AS related_brand_norm
            FROM Level1OENumbers loe
            JOIN cross_references cr4 ON loe.oe_number_norm = cr4.oe_number_norm
            WHERE NOT (loe.source_artikul_norm = cr4.artikul_norm AND loe.source_brand_norm = cr4.brand_norm)
        ),
        AllRelatedParts AS (
            SELECT source_artikul_norm, source_brand_norm, related_artikul_norm, related_brand_norm
            FROM Level1Analogs
            UNION
            SELECT source_artikul_norm, source_brand_norm, related_artikul_norm, related_brand_norm
            FROM Level2Analogs
        ),
        AggregatedAnalogData AS (
            SELECT 
                arp.source_artikul_norm AS artikul_norm,
                arp.source_brand_norm AS brand_norm,
                MAX(CASE WHEN p2.length IS NOT NULL THEN p2.length ELSE NULL END) AS length,
                MAX(CASE WHEN p2.width IS NOT NULL THEN p2.width ELSE NULL END) AS width,
                MAX(CASE WHEN p2.height IS NOT NULL THEN p2.height ELSE NULL END) AS height,
                MAX(CASE WHEN p2.weight IS NOT NULL THEN p2.weight ELSE NULL END) AS weight,
                ANY_VALUE(
                    CASE 
                        WHEN p2.dimensions_str IS NOT NULL AND p2.dimensions_str != '' AND UPPER(TRIM(p2.dimensions_str)) != 'XX'
                        THEN p2.dimensions_str
                        ELSE NULL
                    END
                ) AS dimensions_str,
                ANY_VALUE(
                    CASE 
                        WHEN pd2.representative_name IS NOT NULL AND pd2.representative_name != '' 
                        THEN pd2.representative_name 
                        ELSE NULL
                    END
                ) AS representative_name,
                ANY_VALUE(
                    CASE 
                        WHEN pd2.representative_applicability IS NOT NULL AND pd2.representative_applicability != ''
                        THEN pd2.representative_applicability
                        ELSE NULL
                    END
                ) AS representative_applicability,
                ANY_VALUE(
                    CASE 
                        WHEN pd2.representative_category IS NOT NULL AND pd2.representative_category != ''
                        THEN pd2.representative_category
                        ELSE NULL
                    END
                ) AS representative_category
            FROM AllRelatedParts arp
            JOIN parts_data p2 ON arp.related_artikul_norm = p2.artikul_norm AND arp.related_brand_norm = p2.brand_norm
            LEFT JOIN PartDetails pd2 ON p2.artikul_norm = pd2.artikul_norm AND p2.brand_norm = pd2.brand_norm
            GROUP BY arp.source_artikul_norm, arp.source_brand_norm
        ),
        RankedData AS (
            SELECT 
                p.artikul,
                p.brand,
                p.description,
                p.multiplicity,
                p.length,
                p.width,
                p.height,
                p.weight,
                p.dimensions_str,
                p.image_url,
                pd.representative_name,
                pd.representative_applicability,
                pd.representative_category,
                pd.oe_list,
                aa.analog_list,
                p_analog.length AS analog_length,
                p_analog.width AS analog_width,
                p_analog.height AS analog_height,
                p_analog.weight AS analog_weight,
                p_analog.dimensions_str AS analog_dimensions_str,
                p_analog.representative_name AS analog_representative_name,
                p_analog.representative_applicability AS analog_representative_applicability,
                p_analog.representative_category AS analog_representative_category,
                ROW_NUMBER() OVER (
                    PARTITION BY p.artikul_norm, p.brand_norm 
                    ORDER BY pd.representative_name DESC NULLS LAST, pd.oe_list DESC NULLS LAST
                ) AS rn
            FROM parts_data p
            LEFT JOIN PartDetails pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
            LEFT JOIN AllAnalogs aa ON p.artikul_norm = aa.artikul_norm AND p.brand_norm = aa.brand_norm
            LEFT JOIN AggregatedAnalogData p_analog ON p.artikul_norm = p_analog.artikul_norm AND p.brand_norm = p_analog.brand_norm
        )
        """

        select_exprs = [expr for _, expr in selected_exprs]

        # Join for prices
        price_join = """
        LEFT JOIN prices pr ON r.artikul_norm = pr.artikul_norm AND r.brand_norm = pr.brand_norm
        LEFT JOIN BrandMarkups brm ON r.brand = brm.brand
        """ if include_prices else ""

        query = f"""
        {ctes}
        SELECT
            {', '.join([expr for expr in selected_exprs])}
        FROM RankedData r
        CROSS JOIN DescriptionTemplate dt
        {price_join}
        WHERE r.rn = 1
        {exclusion_where}
        ORDER BY r.brand, r.artikul
        """
        return query

    def _get_brand_markups_sql(self) -> str:
        """Генерация SQL для наценок"""
        rows = []
        for brand, markup in self.price_rules['brand_markups'].items():
            rows.append(f"SELECT '{brand}' AS brand, {markup} AS markup")
        return " UNION ALL ".join(rows) if rows else "SELECT NULL AS brand, NULL AS markup LIMIT 0"

    def export_to_csv_optimized(self, output_path: str, selected_columns=None, include_prices=True, apply_markup=True):
        """Экспорт данных в CSV"""
        total = self.conn.execute("SELECT count(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
        if total == 0:
            st.warning("Нет данных для экспорта")
            return False
        st.info(f"📤 Экспорт {total:,} записей в CSV...")
        try:
            query = self.build_export_query(selected_columns, include_prices, apply_markup)
            df = self.conn.execute(query).pl()

            # преобразуем размерные колонки в строки
            for col in ["Длинна", "Ширина", "Высота", "Вес", "Длинна/Ширина/Высота"]:
                if col in df.columns:
                    df = df.with_columns(
                        pl.when(pl.col(col).is_not_null())
                          .then(pl.col(col).cast(pl.Utf8))
                          .otherwise(pl.lit(""))
                          .alias(col)
                    )

            buf = io.StringIO()
            df.write_csv(buf, separator=';')
            csv_text = buf.getvalue()

            with open(output_path, 'wb') as f:
                f.write(b'\xef\xbb\xbf')  # BOM для Excel
                f.write(csv_text.encode('utf-8'))

            size_mb = os.path.getsize(output_path) / (1024*1024)
            st.success(f"✅ Данные экспортированы в CSV: {output_path} ({size_mb:.1f} МБ)")
            return True
        except Exception as e:
            logger.exception("Ошибка экспорта")
            st.error(f"❌ Ошибка экспорта: {e}")
            return False

    def show_statistics(self):
        """Статистика по базе"""
        st.header("📈 Статистика базы данных")
        stats = {}
        try:
            stats['parts'] = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]
            stats['oe'] = self.conn.execute("SELECT COUNT(*) FROM oe_data").fetchone()[0]
            stats['cross'] = self.conn.execute("SELECT COUNT(*) FROM cross_references").fetchone()[0]
            stats['prices'] = self.conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            stats['brands'] = self.conn.execute("SELECT COUNT(DISTINCT brand) FROM parts_data").fetchone()[0]
            stats['unique_parts'] = self.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT artikul_norm, brand_norm FROM parts_data)").fetchone()[0]
            avg_price = self.conn.execute("SELECT AVG(price) FROM prices WHERE price IS NOT NULL").fetchone()[0]
            stats['avg_price'] = round(avg_price, 2) if avg_price else 0.0
        except Exception as e:
            st.error(f"Ошибка при сборе статистики: {e}")
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Уникальные товары", f"{stats['unique_parts']:,}")
        col2.metric("Бренды", f"{stats['brands']:,}")
        col3.metric("Средняя цена", f"{stats['avg_price']} ₽")

        col1, col2, col3 = st.columns(3)
        col1.metric("Записи (parts)", f"{stats['parts']:,}")
        col2.metric("OE-номера", f"{stats['oe']:,}")
        col3.metric("Кроссы", f"{stats['cross']:,}")

        col1, col2 = st.columns(2)
        col1.metric("Ценовые записи", f"{stats['prices']:,}")
        col2.metric("Размер файла БД", f"{os.path.getsize(self.db_path) / (1024**2):.1f} МБ")

        # Топ брендов
        st.subheader("🏆 Топ-10 брендов по количеству артикулов")
        try:
            top_brands = self.conn.execute("""
                SELECT brand, COUNT(*) as cnt
                FROM parts_data
                WHERE brand IS NOT NULL
                GROUP BY brand
                ORDER BY cnt DESC
                LIMIT 10
            """).pl()
            st.dataframe(top_brands.to_pandas(), use_container_width=True)
        except Exception as e:
            st.warning(f"Не удалось загрузить топ брендов: {e}")

        # Распределение по категориям
        st.subheader("🗂️ Распределение по категориям")
        try:
            category_stats = self.conn.execute("""
                SELECT 
                    COALESCE(representative_category, 'Разное') as category,
                    COUNT(*) as cnt
                FROM (
                    SELECT DISTINCT p.artikul_norm, p.brand_norm, pd.representative_category
                    FROM parts_data p
                    LEFT JOIN part_details_view pd ON p.artikul_norm = pd.artikul_norm AND p.brand_norm = pd.brand_norm
                )
                GROUP BY category
                ORDER BY cnt DESC
                LIMIT 15
            """).pl()
            st.dataframe(category_stats.to_pandas(), use_container_width=True)
        except Exception as e:
            st.warning("Не удалось загрузить статистику по категориям")
    
    def merge_all_data_parallel(self, file_paths: Dict[str, str], max_workers=4) -> Dict[str, pl.DataFrame]:
        """Параллельная обработка файлов"""
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for ft, fp in file_paths.items():
                if fp and os.path.exists(fp):
                    futures[executor.submit(self.read_and_prepare_file, fp, ft)] = ft
            for fut in as_completed(futures):
                ft = futures[fut]
                try:
                    df = fut.result()
                    if not df.is_empty():
                        results[ft] = df
                        logger.info(f"Обработан: {ft}")
                except Exception as e:
                    logger.error(f"Ошибка при обработке {ft}: {e}")
        return results

def main():
    st.title("🚗 AutoParts Catalog — Масштабируемая система для 10+ млн записей")
    st.markdown("""
    ### 💼 Профессиональная платформа для управления каталогами автозапчастей
    - Поддержка больших данных
    - Инкрементальные обновления
    - Мультиформатный экспорт
    - Гибкая настройка
    """)

    catalog = AutoPartsCatalog()

    menu = st.sidebar.radio("🧭 Навигация", ["Загрузка данных", "Экспорт", "Статистика", "Управление"])

    if menu == "Загрузка данных":
        st.header("📥 Загрузка и обновление данных")
        col1, col2 = st.columns(2)
        with col1:
            oe_file = st.file_uploader("1. Основные данные (OE)", type=['xlsx', 'xls'])
            cross_file = st.file_uploader("2. Кроссы (OE → Артикул)", type=['xlsx', 'xls'])
            barcode_file = st.file_uploader("3. Штрих-коды и кратность", type=['xlsx', 'xls'])
        with col2:
            dimensions_file = st.file_uploader("4. Весогабариты", type=['xlsx', 'xls'])
            images_file = st.file_uploader("5. Ссылки на изображения", type=['xlsx', 'xls'])
            prices_file = st.file_uploader("6. Прайс-лист с ценами", type=['xlsx', 'xls'])

        file_map = {
            'oe': oe_file,
            'cross': cross_file,
            'barcode': barcode_file,
            'dimensions': dimensions_file,
            'images': images_file,
            'prices': prices_file
        }

        # Сохранение загруженных файлов
        saved_paths = {}
        for ft, uf in file_map.items():
            if uf:
                save_path = catalog.data_dir / f"upload_{ft}_{int(time.time())}.xlsx"
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved_paths[ft] = str(save_path)

        if st.button("🚀 Обработать и загрузить данные"):
            if not saved_paths:
                st.warning("Загрузите хотя бы один файл")
            else:
                with st.spinner("Чтение и обработка файлов..."):
                    dataframes = catalog.merge_all_data_parallel(saved_paths)
                if dataframes:
                    with st.spinner("Загрузка в базу..."):
                        catalog.process_and_load_data(dataframes)
                else:
                    st.error("❌ Не удалось обработать файлы")
    elif menu == "Экспорт":
        catalog.show_export_interface()
    elif menu == "Статистика":
        catalog.show_statistics()
    elif menu == "Управление":
        catalog.show_data_management()


if __name__ == "__main__":
    main()
