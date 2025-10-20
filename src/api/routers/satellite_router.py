from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
import io, json, os


router = APIRouter()
templates = Jinja2Templates(directory="templates")

model = nn.Sequential(
    # Первый сверточный блок - аналог Conv2D + MaxPooling2D в TF
    nn.Conv2d(3, 16, 3, padding=1),  # 3 входных канала (RGB), 16 выходных, ядро 3x3
    nn.BatchNorm2d(16),
    nn.LeakyReLU(),  # Функция активации как в TF
    nn.MaxPool2d(2),  # Пулинг 2x2 как MaxPooling2D() в TF

    # Второй сверточный блок
    nn.Conv2d(16, 32, 3, padding=1),  # 16 входных, 32 выходных канала
    nn.BatchNorm2d(32),
    nn.LeakyReLU(),
    nn.MaxPool2d(2),

    # Третий сверточный блок
    nn.Conv2d(32, 64, 3, padding=1),  # 32 входных, 64 выходных канала
    nn.BatchNorm2d(64),
    nn.LeakyReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, 3, padding=1),  # 64 входных, 128 выходных канала
    nn.BatchNorm2d(128),
    nn.LeakyReLU(),
    nn.MaxPool2d(2),

    # Полносвязные слои - аналог Dense в TF
    nn.Flatten(),  # Вытягиваем в вектор как Flatten() в TF
    nn.Dropout(0.2),  # Dropout такой же как в TF
    nn.Linear(128 * 14 * 14, 256),  # Полносвязный слой: 128*14*14 -> 128 нейронов
    nn.LeakyReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 5)  # Выходной слой: 128 -> количество классов
)

model_path = "src/models/satellite_model_pytorch.pth"
classes_path = "src/models/satellite_classes_pytorch.json"

if os.path.exists(model_path) and os.path.exists(classes_path):
    try:
        # Загрузка классов - полностью идентично TF
        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)

        # ОТЛИЧИЕ OT TENSORFLOW: нужно создать модель и загрузить веса
        # Обновляем выходной слой под правильное количество классов
        model[-1] = nn.Linear(256, len(class_names))  # Заменяем последний слой

        # Загружаем веса - аналог load_model() в TF
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Важно: переводим модель в режим оценки (как inference в TF)

        print("PyTorch модель цветов загружена успешно")
        print("Доступные классы:", class_names)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        model = None
        class_names = []
else:
    print("PyTorch модель не найдена. Запустите train_flowers_pytorch.py для обучения модели.")
    model = None
    class_names = []

# ТРАНСФОРМАЦИИ ДЛЯ ПРЕДОБРАБОТКИ - аналог предобработки в TF
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера как в TF
    transforms.ToTensor(),  # Конвертация в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@router.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("satellite_form.html", {
        "request": request,
        "classes": class_names,
        "title": "Распознавание спутниковых снимков"
    })

@router.post("/predict", response_class=HTMLResponse)
async def predict_display(
        request: Request,
        file: UploadFile = File(...)
    ):
    try:
        # ЧТЕНИЕ ИЗОБРАЖЕНИЯ - полностью идентично TF версии
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        # СОХРАНЕНИЕ ДЛЯ ОТОБРАЖЕНИЯ - без изменений
        uploads_dir = "static/uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        img_path = f"{uploads_dir}/{file.filename}"
        img.save(img_path)

        # ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЯ - концепт такой же но другой синтаксис
        input_tensor = transform(img)  # Применяем трансформации
        input_batch = input_tensor.unsqueeze(0)  # Добавляем batch dimension (аналог expand_dims в TF)

        # ПРЕДСКАЗАНИЕ - концепт такой же как model.predict() в TF
        with torch.no_grad():  # Отключаем вычисление градиентов для inference
            output = model(input_batch)  # Прямой проход через модель

        # ОБРАБОТКА РЕЗУЛЬТАТОВ - похоже на TF но с PyTorch синтаксисом
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Softmax как в TF
        confidence, predicted_class = torch.max(probabilities, 0)  # Берем максимальную вероятность

        # Конвертируем в Python числа - аналог .numpy() в TF
        confidence = confidence.item()  # Получаем значение уверенности
        predicted_class = predicted_class.item()  # Получаем индекс предсказанного класса

        # Получаем все вероятности для отображения
        all_probs = probabilities.numpy()  # Конвертируем в numpy массив

        return templates.TemplateResponse("satellite_result.html", {
            "request": request,
            "predicted_class": predicted_class,
            "classes": class_names[predicted_class],
            "confidence": round(confidence * 100, 2),
            "image_url": f"/{img_path}",
            "all_predictions": [{"class": i, "name": name, "prob": round(prob * 100, 2)}
                                for i, (name, prob) in enumerate(zip(class_names, all_probs))]
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e), "status_code": 500})