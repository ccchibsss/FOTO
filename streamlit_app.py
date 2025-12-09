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
import warnings
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXCEL_ROW_LIMIT = 1_048_576

class HighVolumeAutoPartsCatalog:
    def __init__(self):
        self.data_dir = Path("./auto_parts_data")
        self.data_dir.mkdir(exist_ok=True)

        self.cloud_config = self.load_cloud_config()
        self.db_path = self.data_dir / "catalog.duckdb"
        self.conn = duckdb.connect(database=str(self.db_path))
        self.setup_database()

        # Новое: хранение информации о всех колонках
        self.table_columns = {
            'oe_data': set(),
            'parts_data': set(),
            'cross_references': set(),
            'prices': set()
        }

        # Загрузка текущей структуры таблиц
        self.load_table_structure()

        # Загрузка правил
        self.price_rules = self.load_price_rules()
        self.exclusion_rules = self.load_exclusion_rules()
        self.category_mapping = self.load_category_mapping()

        # UI настройки
        st.set_page_config(
            page_title="🚗 AutoParts Catalog",
            layout="wide",
            page_icon="🚗",
            initial_sidebar_state="expanded"
        )

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
        config_path.write_text(json.dumps(default_config, indent=2, ensure_ascii=False), encoding='utf-8')
        return default_config

    def save_cloud_config(self):
        config_path = self.data_dir / "cloud_config.json"
        self.cloud_config["last_sync"] = int(time.time())
        config_path.write_text(json.dumps(self.cloud_config, indent=2, ensure_ascii=False), encoding='utf-8')

    def load_price_rules(self) -> Dict[str, Any]:
        path = self.data_dir / "price_rules.json"
        default = {
            "global_markup": 0.2,
            "brand_markups": {},
            "min_price": 0.0,
            "max_price": 99999.0
        }
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
        default = {
            "Радиатор": "Охлаждение",
            "Шаровая опора": "Подвеска",
            "Фильтр масляный": "Фильтры",
            "Тормозные колодки": "Тормоза"
        }
        if path.exists():
            try:
                mapping = {}
                for line in path.read_text(encoding='utf-8').splitlines():
                    if line.strip() and "|" in line:
                        k, v = line.split("|", 1)
                        mapping[k.strip()] = v.strip()
                return mapping
            except Exception as e:
                logger.error(f"Ошибка: {e}")
                return default
        content = "\n".join([f"{k}|{v}" for k, v in default.items()])
        path.write_text(content, encoding='utf-8')
        return default

    def save_category_mapping(self):
        path = self.data_dir / "category_mapping.txt"
        content = "\n".join([f"{k}|{v}" for k, v in self.category_mapping.items()])
        path.write_text(content, encoding='utf-8')

    def setup_database(self):
        # Создаем таблицы, если их еще нет
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS oe_data (
                oe_number_norm VARCHAR PRIMARY KEY,
                oe_number VARCHAR,
                name VARCHAR,
                applicability VARCHAR
            )
        """)
        self.conn.execute("""
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
        self.create_indexes()

    def create_indexes(self):
        """Создаем индексы для ускорения поиска"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_oe_data_oe ON oe_data(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_parts_data_keys ON parts_data(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_oe ON cross_references(oe_number_norm)",
            "CREATE INDEX IF NOT EXISTS idx_cross_artikul ON cross_references(artikul_norm, brand_norm)",
            "CREATE INDEX IF NOT EXISTS idx_prices_keys ON prices(artikul_norm, brand_norm)"
        ]
        for sql in indexes:
            self.conn.execute(sql)

    def load_table_structure(self):
        """Загружаем текущий список колонок таблиц для дальнейшего расширения"""
        for table in self.table_columns.keys():
            res = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            cols = {row[1] for row in res}
            self.table_columns[table] = cols

    def add_new_column(self, table_name: str, column_name: str, col_type: str = "VARCHAR"):
        """Добавление нового столбца в таблицу, если его еще нет"""
        if column_name not in self.table_columns[table_name]:
            try:
                self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {col_type}")
                self.table_columns[table_name].add(column_name)
                logger.info(f"Добавлен столбец {column_name} в таблицу {table_name}")
            except Exception as e:
                logger.error(f"Ошибка при добавлении столбца {column_name} в {table_name}: {e}")

    def update_table_structure_with_df(self, table_name: str, df: pl.DataFrame):
        """Обновление структуры таблицы в базе по новым колонкам из DataFrame"""
        existing_cols = self.table_columns.get(table_name, set())
        for col in df.columns:
            if col not in existing_cols:
                self.add_new_column(table_name, col)

    def read_and_prepare_file(self, file_path: str, file_type: str) -> pl.DataFrame:
        """Чтение файла и подготовка"""
        logger.info(f"📄 Обработка файла: {file_type} ({file_path})")
        try:
            df = pl.read_excel(file_path, engine='calamine')
        except Exception as e:
            logger.exception(f"❌ Ошибка чтения файла {file_path}: {e}")
            return pl.DataFrame()

        # Определение ожидаемых колонок
        schemas = {
            'oe': ['oe_number', 'artikul', 'brand', 'name', 'applicability'],
            'barcode': ['brand', 'artikul', 'barcode', 'multiplicity'],
            'dimensions': ['artikul', 'brand', 'length', 'width', 'height', 'weight', 'dimensions_str'],
            'images': ['artikul', 'brand', 'image_url'],
            'cross': ['oe_number', 'artikul', 'brand'],
            'prices': ['artikul', 'brand', 'price', 'currency']
        }
        expected_cols = schemas.get(file_type, [])
        column_mapping = self.detect_columns(df.columns, expected_cols)

        # Переименование колонок
        if column_mapping:
            df = df.rename(column_mapping)
        else:
            logger.warning(f"⚠️ Не удалось определить колонки для типа {file_type}")

        # Очистка и нормализация
        for col in ['artikul', 'brand', 'oe_number']:
            if col in df.columns:
                df = df.with_columns(self.clean_values(pl.col(col)).alias(col))
                df = df.with_columns(self.normalize_key(pl.col(col)).alias(f"{col}_norm"))

        df = df.unique()

        # Обновление структуры таблиц
        table_name = self.get_table_name_by_type(file_type)
        if table_name:
            self.update_table_structure_with_df(table_name, df)

        return df

    def get_table_name_by_type(self, file_type: str) -> Optional[str]:
        """Определение таблицы по типу файла"""
        mapping = {
            'oe': 'oe_data',
            'cross': 'cross_references',
            'barcode': 'parts_data',
            'dimensions': 'parts_data',
            'images': 'parts_data',
            'prices': 'prices'
        }
        return mapping.get(file_type)

    def detect_columns(self, actual_columns: List[str], expected_columns: List[str]) -> Dict[str, str]:
        """Автоматическое определение колонок по ключевым словам"""
        variants = {
            'oe_number': ['oe номер', 'oe', 'оe', 'номер', 'code'],
            'artikul': ['артикул', 'article', 'sku'],
            'brand': ['бренд', 'brand', 'производитель'],
            'name': ['наименование', 'название', 'name', 'описание'],
            'applicability': ['применимость', 'автомобиль', 'vehicle'],
            'barcode': ['штрих-код', 'barcode', 'ean'],
            'multiplicity': ['кратность шт', 'кратность', 'multiplicity'],
            'length': ['длина (см)', 'длина', 'length'],
            'width': ['ширина (см)', 'ширина', 'width'],
            'height': ['высота (см)', 'высота', 'height'],
            'weight': ['вес (кг)', 'вес', 'weight'],
            'image_url': ['ссылка', 'url', 'изображение', 'image'],
            'dimensions_str': ['весогабариты', 'размеры', 'dimensions'],
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

    def clean_values(self, col: pl.Expr) -> pl.Expr:
        """Очистка строк"""
        return pl.when(pl.col(col).is_null()).then("").otherwise(
            pl.col(col).cast(pl.Utf8).str.replace_all("'", "").str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\\-\\s]", "").str.strip()
        )

    def normalize_key(self, col: pl.Expr) -> pl.Expr:
        """Нормализация ключей"""
        return pl.when(pl.col(col).is_null()).then("").otherwise(
            pl.col(col).cast(pl.Utf8).str.replace_all("'", "").str.replace_all(r"[^0-9A-Za-zА-Яа-яЁё`\\-\\s]", "").str.strip().str.to_lowercase()
        )

    def get_table_name_by_type(self, file_type: str) -> Optional[str]:
        mapping = {
            'oe': 'oe_data',
            'cross': 'cross_references',
            'barcode': 'parts_data',
            'dimensions': 'parts_data',
            'images': 'parts_data',
            'prices': 'prices'
        }
        return mapping.get(file_type)

    def upsert_data(self, table_name: str, df: pl.DataFrame, pk: List[str]):
        """Обновление или вставка данных, с расширением таблицы"""
        if df.is_empty():
            return
        # Обновляем структуру таблицы
        self.update_table_structure_with_df(table_name, df)

        # Конвертируем DataFrame в Arrow для загрузки
        self.conn.register("temp_df", df.to_arrow())
        pk_str = ", ".join(f'"{col}"' for col in pk)
        update_cols = [col for col in df.columns if col not in pk]
        if update_cols:
            set_clause = ", ".join([f'"{col}" = excluded."{col}"' for col in update_cols])
            sql = f"""
                INSERT INTO {table_name} SELECT * FROM temp_df
                ON CONFLICT ({pk_str}) DO UPDATE SET {set_clause}
            """
        else:
            sql = f"INSERT INTO {table_name} SELECT * FROM temp_df ON CONFLICT ({pk_str}) DO NOTHING"
        try:
            self.conn.execute(sql)
        finally:
            self.conn.unregister("temp_df")

    def update_table_structure_with_df(self, table_name: str, df: pl.DataFrame):
        """Добавление новых колонок в таблицы по мере необходимости"""
        existing_cols = self.table_columns.get(table_name, set())
        for col in df.columns:
            if col not in existing_cols:
                self.add_new_column(table_name, col, col_type="VARCHAR")
                existing_cols.add(col)

    def add_new_column(self, table_name: str, column_name: str, col_type: str = "VARCHAR"):
        """Добавление нового столбца"""
        try:
            self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {col_type}")
            self.table_columns[table_name].add(column_name)
        except Exception as e:
            logger.error(f"Ошибка при добавлении столбца {column_name} в {table_name}: {e}")

    def build_export_query(self, selected_columns=None, include_prices=True, apply_markup=True) -> str:
        """Построение SQL-запроса для экспорта, с учетом выбранных колонок"""
        cols = selected_columns or [
            "Артикул", "Бренд", "Наименование", "Применимость", "Кратность", "Штрих-код", "Ссылка на изображение", "аналоги"
        ]
        # Собираем SELECT
        select_parts = []
        select_parts.append("p.artikul AS \"Артикул\"")
        select_parts.append("p.brand AS \"Бренд\"")
        select_parts.append("COALESCE(od.name, 'Не указано') AS \"Наименование\"")
        select_parts.append("COALESCE(od.applicability, 'Для всех') AS \"Применимость\"")
        select_parts.append("p.multiplicity AS \"Кратность\"")
        select_parts.append("p.barcode AS \"Штрих-код\"")
        select_parts.append("p.image_url AS \"Ссылка на изображение\"")
        select_parts.append("STRING_AGG(DISTINCT cr2.artikul, ', ') AS \"аналоги\"")
        # Можно добавить дополнительные поля по необходимости

        query = f"""
        SELECT
            {', '.join(select_parts)}
        FROM parts_data p
        LEFT JOIN cross_references cr ON p.artikul_norm = cr.artikul_norm AND p.brand_norm = cr.brand_norm
        LEFT JOIN oe_data od ON cr.oe_number_norm = od.oe_number_norm
        LEFT JOIN cross_references cr2 ON cr.oe_number_norm = cr2.oe_number_norm
        GROUP BY p.artikul, p.brand, od.name, od.applicability, p.multiplicity, p.barcode, p.image_url
        """

        return query

    def export_to_csv_optimized(self, output_path: str, selected_columns=None, include_prices=True, apply_markup=True):
        """Экспорт данных в CSV"""
        query = self.build_export_query(selected_columns, include_prices, apply_markup)
        df = self.conn.execute(query).pl()

        # Фильтрация по исключениям
        for exclude_word in self.exclusion_rules:
            df = df.filter(~pl.col('Наименование').str.contains(exclude_word))
        # Применение наценки
        if include_prices and apply_markup:
            df = df.with_columns(
                pl.when(pl.col('Бренд').is_not_null())
                .then(
                    pl.col('Цена').apply(lambda price, brand: self.apply_markups(price, brand))
                )
                .otherwise(pl.col('Цена'))
                .alias('Цена с наценкой')
            )
        df.write_csv(output_path, separator=";", include_header=True)
        st.success(f"✅ Экспорт завершен: {output_path}")

    def apply_markups(self, price, brand):
        """Применение наценки"""
        markup = self.price_rules['brand_markups'].get(brand, self.price_rules['global_markup'])
        return price * (1 + markup)

    def show_ui_for_new_columns(self):
        """Интерфейс для добавления новых колонок и данных"""
        st.markdown("## 🔧 Добавление новых данных и колонок")
        table = st.selectbox("Таблица", list(self.table_columns.keys()))
        col_name = st.text_input("Название нового столбца")
        col_type = st.selectbox("Тип данных", ["VARCHAR", "DOUBLE", "INTEGER"])
        if st.button("Добавить колонку"):
            if col_name:
                self.add_new_column(table, col_name, col_type)
                st.success(f"Добавлен столбец {col_name} в таблицу {table}")

        uploaded_file = st.file_uploader("Загрузить файл с данными для добавления", type=["xlsx"])
        if uploaded_file:
            df = self.read_and_prepare_file(uploaded_file, 'custom')
            if not df.is_empty():
                self.update_table_structure_with_df(table, df)
                self.upsert_data(table, df, pk=[])  # Можно задать ключи по необходимости
                st.success("Данные успешно добавлены")

    def show_interface(self):
        """Главный интерфейс"""
        st.title("🚗 AutoParts Catalog")
        menu = st.sidebar.radio("Меню", ["Загрузка", "Экспорт", "Статистика", "Настройки", "Управление структурой"])
        if menu == "Загрузка":
            self.show_upload_ui()
        elif menu == "Экспорт":
            self.show_export_ui()
        elif menu == "Статистика":
            self.show_statistics()
        elif menu == "Настройки":
            self.show_settings()
        elif menu == "Управление структурой":
            self.show_ui_for_new_columns()

    def show_upload_ui(self):
        """UI для загрузки файлов"""
        uploaded_files = {}
        for label, key in [("Основные данные (OE)", "oe"),
                           ("Кроссы", "cross"),
                           ("Штрих-коды", "barcode"),
                           ("Весогабариты", "dimensions"),
                           ("Изображения", "images"),
                           ("Цены", "prices")]:
            uploaded_files[key] = st.file_uploader(label, type=["xlsx"])
        if st.button("Загрузить файлы"):
            for key, file in uploaded_files.items():
                if file:
                    path = self.data_dir / f"upload_{key}_{int(time.time())}.xlsx"
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    df = self.read_and_prepare_file(str(path), key)
                    if not df.is_empty():
                        table_name = self.get_table_name_by_type(key)
                        if table_name:
                            self.update_table_structure_with_df(table_name, df)
                            self.upsert_data(table_name, df, pk=[])
            st.success("Данные загружены и обновлены.")

    def show_export_ui(self):
        """UI для экспорта"""
        self.show_export_interface()

    def show_statistics(self):
        """Показ статистики"""
        total_parts = self.conn.execute("SELECT COUNT(*) FROM parts_data").fetchone()[0]
        st.metric("Всего товаров", total_parts)

    def show_settings(self):
        """Настройки"""
        st.markdown("Настройки — в разработке")

# В основном запуске
def main():
    st.title("🚗 AutoParts Catalog")
    catalog = HighVolumeAutoPartsCatalog()
    catalog.show_interface()

if __name__ == "__main__":
    main()
