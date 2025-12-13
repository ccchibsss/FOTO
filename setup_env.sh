#!/bin/bash

# Этот скрипт автоматически создаст виртуальное окружение с Python 3.11,
# активирует его, обновит pip и установит все зависимости из requirements.txt

# Проверка наличия Python 3.11
if ! command -v python3.11 &> /dev/null
then
    echo "Python 3.11 не установлен. Установите его перед запуском этого скрипта."
    exit 1
fi

# Создаем виртуальное окружение
echo "Создаю виртуальное окружение..."
python3.11 -m venv venv

# Активируем окружение
source venv/bin/activate

# Обновляем pip
echo "Обновляю pip..."
pip install --upgrade pip

# Устанавливаем зависимости из requirements.txt
echo "Устанавливаю зависимости..."
pip install -r requirements.txt

# Установка detectron2
echo "Устанавливаю detectron2..."
pip install --upgrade pip
pip install git+https://github.com/facebookresearch/detectron2.git@main

echo "Настройка окружения завершена."
