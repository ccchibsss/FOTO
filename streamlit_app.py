import streamlit as st
import os
import pandas as pd
import io
from PIL import Image
import glob
import rembg

# Импорт библиотек Google API для работы с Google Drive
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Области доступа для работы с Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Функция авторизации и получения сервиса Google Drive
@st.cache(allow_output_mutation=True)
def get_drive_service():
    """
    Проверяет наличие файла токена, авторизуется или использует существующий.
    Возвращает объект сервиса Google Drive.
    """
    creds = None
    # Проверка существования файла токена
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # Если токена нет или он недействителен - авторизация через браузер
    if not creds or not creds.valid:
        # Запуск OAuth flow
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        # Сохраняем токен для будущего использования
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    # Создаем сервис Google Drive
    service = build('drive', 'v3', credentials=creds)
    return service

# Функция загрузки изображения в папку Google Drive
def upload_image_to_drive(service, folder_id, image, filename):
    """
    Загружает изображение в указанную папку на Google Drive.
    Делает файл публичным и возвращает публичную ссылку.
    """
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    media = MediaIoBaseUpload(buffer, mimetype='image/png')
    # Создаем метаданные файла
    file_metadata = {
        'name': filename,
        'parents': [folder_id],
        'mimeType': 'image/png'
    }
    # Загружаем файл
    file = service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
    # Делаем файл публичным для доступа по ссылке
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    service.permissions().create(fileId=file['id'], body=permission).execute()
    # Возвращаем публичную ссылку
    return file.get('webViewLink')

# Функция получения или создания папки на Google Drive по имени
def get_or_create_folder(service, folder_name):
    """
    Ищет папку по названию. Если не найдена, создает новую.
    Возвращает ID папки.
    """
    query = f"name = '{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if items:
        return items[0]['id']
    else:
        # Создаем новую папку
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        return folder.get('id')

# Функция для удаления фона у изображения
def remove_background(image):
    """
    Удаляет фон у изображения с помощью библиотеки rembg.
    Возвращает изображение без фона.
    """
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    result_bytes = rembg.remove(img_bytes)
    result_img = Image.open(io.BytesIO(result_bytes))
    return result_img

# Основное приложение
st.title("Обработка фото товаров с загрузкой в Google Drive")

# Авторизация и создание сервиса Google Drive
service = get_drive_service()

# Ввод пути к папке с изображениями
folder_path = st.text_input("Введите путь к папке с фото товара:")

# Обработка по нажатию кнопки
if st.button("Обработать и загрузить в Google Drive"):
    if folder_path and os.path.exists(folder_path):
        # Получаем список изображений из папки
        images_files = glob.glob(os.path.join(folder_path, "*.*"))
        if not images_files:
            st.write("В папке не найдено изображений.")
        else:
            data = []
            # Получаем или создаем папку на Google Drive по имени папки
            folder_name = os.path.basename(folder_path)
            folder_id = get_or_create_folder(service, folder_name)
            # Обрабатываем каждое изображение
            for path in images_files:
                try:
                    name = os.path.basename(path)
                    # Открываем изображение
                    img = Image.open(path)
                    # Показываем оригинал
                    st.image(img, caption=f"Оригинал: {name}")

                    # Удаление фона
                    img_no_bg = remove_background(img)

                    # Загрузка в Google Drive и получение публичной ссылки
                    link = upload_image_to_drive(service, folder_id, img_no_bg, name)

                    # Добавляем в список данных
                    data.append({"Название": name, "Ссылка": link})

                except Exception as e:
                    st.write(f"Ошибка с файлом {name}: {e}")

            # Создаем DataFrame и Excel файл
            df = pd.DataFrame(data)
            df = df[["Название", "Ссылка"]]
            excel_bytes = io.BytesIO()
            df.to_excel(excel_bytes, index=False)
            excel_bytes.seek(0)

            # Предлагаем скачать Excel
            st.download_button(
                label="Скачать Excel с ссылками",
                data=excel_bytes,
                file_name="links.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.write("Пожалуйста, введите корректный путь к папке.")
