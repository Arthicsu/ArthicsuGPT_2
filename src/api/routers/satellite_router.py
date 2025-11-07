from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

from PIL import Image
import numpy as np
import io, json, os


router = APIRouter()
templates = Jinja2Templates(directory="templates")

model_path = "src/models/satellite_model_MobileNetV2_transfer.h5"
classes_path = "src/models/satellite_classes.json"

if os.path.exists(model_path) and os.path.exists(classes_path):
    try:
        model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        class_names = []
else:
    print("Модель не найдена")
    class_names = []

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
        # Чтение и предобработка изображения
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        # Изменение размера до 224x224 (как требует MobileNetV2)
        img = img.resize((224, 224))

        # Конвертируем в массив numpy
        img_array = image.img_to_array(img)

        # Предобработка для MobileNetV2
        img_array = preprocess_input(img_array)

        # Добавляем размерности: (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Предсказание
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Сохранение загруженного изображения для отображения
        uploads_dir = "static/uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        img_path = f"{uploads_dir}/{file.filename}"
        img.save(img_path)

        # Подготовка данных для отображения
        all_predictions = []
        for i, name in enumerate(class_names):
            all_predictions.append({
                "class": i,
                "name": name,
                "prob": round(prediction[0][i] * 100, 2)
            })

        # Сортируем по убыванию вероятности для лучшего отображения
        all_predictions.sort(key=lambda x: x["prob"], reverse=True)

        return templates.TemplateResponse("satellite_result.html", {
            "request": request,
            "predicted_class": predicted_class,
            "class_name": class_names[predicted_class],
            "confidence": round(confidence * 100, 2),
            "image_url": f"/{img_path}",
            "all_predictions": all_predictions
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e), "status_code": 500})