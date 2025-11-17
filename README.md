# OpenCV-FastAPI - Детектирование объектов на изображении
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-red)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119%2B-green)]()
<br>
[![ssd_mobilenet_v2](https://img.shields.io/badge/MobileNetV2-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## Краткое описание
OpenCV-FastAPI — это высокопроизводительное веб-приложение для детектирования объектов на изображении на основе предобученной SSD модели (ssd_mobilenet_v2).

## Особенности
- **Асинхронная обработка**: Оптимизированная работа с изображениями
- **Использование SSD модели**: [ssd_mobilenet_v2](https://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz)
- **Датасет COCO17 (320x320)**

## Установка
> Для работы приложения требуется Python 3.11
- Клонируйте репозиторий в вашу среду разработки:
	```
	git clone https://github.com/Arthicsu/ArthicsuGPT_2.git
	```
- Установите все необходимые модули и библиотеки:
	```
	pip install -r requirements.txt
	```
- Скачайте последнюю версию SSD модели. Выше ссылка в разделе **Особенности**

- Запустите приложение:
	```
	uvicorn src.main:app --reload
	```

## Использование
1. Перейдите по адресу `http://localhost:8000`.
2. Отправьте изображение и нажмите "Распознать изображение".

## Лицензия
- Этот проект распространяется под лицензией MIT.