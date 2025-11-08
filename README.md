# VGG16-FastAPI - Классификация спутниковых снимков
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-red)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119%2B-green)]()
<br>
[![VGG16](https://img.shields.io/badge/VGG16-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
<div align="center"><img width="554" height="506" alt="image" class="idk" src="https://github.com/user-attachments/assets/b921f833-7181-4615-a74e-1b155642ebc2" /></div>

## Краткое описание
- VGG16-FastAPI — это высокопроизводительное веб-приложение для классификации спутниковых снимков на основе предобученной модели VGG16.
- Приложение использует предобученную модель VGG16, которая теперь классифицирует спутниковые снимки.

## Особенности
- **Асинхронная обработка**: Оптимизированная работа с изображениями
- **Я долго модель дообучал** - 20 эпох как никак

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
- Запустите приложение:
	```
	uvicorn src.main:app --reload
	```

## Использование
1. Перейдите по адресу `http://localhost:8000`.
2. Отправьте изображение и нажмите "Распознать изображение".

## Примеры классификации
- Пылевые бури
- Море (вода)
- Суша
- Пустыня


## Лицензия
- Этот проект распространяется под лицензией MIT.