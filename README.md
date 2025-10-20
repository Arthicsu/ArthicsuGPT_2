[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-red)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119%2B-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
# PyTorch-FastAPI - Классификация спутниковых снимков

<div align="center"><img width="554" height="506" alt="image" class="alig" src="https://github.com/user-attachments/assets/b921f833-7181-4615-a74e-1b155642ebc2" /></div>

## Краткое описание
- PyTorch-FastAPI — это высокопроизводительное веб-приложение для классификации спутниковых снимков на основе PyTorch.
- Приложение использует сверточную нейронную сеть (CNN), обученную на PyTorch, для точного определения категорий спутниковых снимков.

## Особенности
- **Асинхронная обработка**: Оптимизированная работа с изображениями



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