from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from keras.models import load_model
from keras.preprocessing import image

from PIL import Image
import numpy as np
import io, json, os

router = APIRouter()

templates = Jinja2Templates(directory="templates")

model_path = "src/models/satellite_model.h5"
classes_path = "src/models/satellite_classes.json"
if os.path.exists(model_path) and os.path.exists(classes_path):
    try:
        model = load_model(model_path)
        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print("Модель цветов загружена успешно")
        print("Доступные классы:", class_names)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        model = None
        class_names = []
else:
    print("Модель не найдена. Запустите train.py для обучения модели.")
    model = None
    class_names = []


@router.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("satellite_form.html", {
        "request": request,
        "classes": class_names,
        "model_loaded": model is not None,
        "title": "Распознавание спутниковых снимков"
    })

@router.post("/predict", response_class=HTMLResponse)
async def predict_display(
        request: Request,
        file: UploadFile = File(...)
    ):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        img = img.resize((180, 180))

        # Конвертируем в массив numpy
        img_array = image.img_to_array(img)

        # Добавляем размерности batch, размерности фото, каналы цветов: (1, 180, 180, 3)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        uploads_dir = "static/uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        img_path = f"{uploads_dir}/{file.filename}"
        img.save(img_path)

        return templates.TemplateResponse("satellite_result.html", {
            "request": request,
            "predicted_class": predicted_class,
            "class_name": class_names[predicted_class],
            "confidence": round(confidence * 100, 2),
            "image_url": f"/{img_path}",
            "all_predictions": [{"class": i, "name": name, "prob": round(prediction[0][i] * 100, 2)}
                                for i, name in enumerate(class_names)]
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e), "status_code": 500})