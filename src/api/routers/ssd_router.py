from fastapi import APIRouter, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import tensorflow as tf

from PIL import Image
import numpy as np
import io, json, os


router = APIRouter()
templates = Jinja2Templates(directory="templates")

model_path = "src/models/ssd_mobilenet_v2/saved_model"

coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class_colors = {
    'person': (255, 0, 0),  # Синий в BGR
    'car': (0, 255, 0),  # Зеленый в BGR
    'dog': (0, 0, 255),  # Красный в BGR
    'cat': (255, 255, 0),  # Голубой в BGR
    'bird': (255, 0, 255),  # Розовый в BGR
    'chair': (0, 255, 255),  # Желтый в BGR
    'default': (128, 128, 128)  # Серый в BGR
}

def get_color_for_class(class_name):
    return class_colors.get(class_name, class_colors['default'])

try:
    model = tf.saved_model.load(model_path)
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")


@router.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("ssd_form.html", {
        "request": request,
        "classes": coco_classes,
        "default_threshold": 50,
        "title": "Распознавание спутниковых снимков"
    })

detection_history = []
@router.post("/predict", response_class=HTMLResponse)
async def predict_display(
        request: Request,
        file: UploadFile = File(...),
        threshold: float = Form(50.0),
        selected_classes: str = Form("")
    ):
    try:
        confidence_threshold = threshold / 100.0

        if selected_classes:
            selected_classes_list = [cls.strip().lower() for cls in selected_classes.split(",")]
            valid_classes = [cls for cls in selected_classes_list if cls in [c.lower() for c in coco_classes]]
        else:
            valid_classes = []

        # Читаем загруженное изображение
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ВАЖНО: Сохраняем оригинал в BGR для отображения
        original_image = image.copy()

        # Для модели конвертируем BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Подготавливаем изображение для модели
        input_tensor = tf.convert_to_tensor(image_rgb)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Выполняем детектирование
        detections = model(input_tensor)

        # Обрабатываем результаты
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()

        # Фильтруем результаты
        detected_objects = []
        height, width = image.shape[:2]

        for i in range(len(scores)):
            if scores[i] > confidence_threshold:
                class_id = int(classes[i])
                class_name = coco_classes[class_id - 1]

                if valid_classes and class_name.lower() not in valid_classes:
                    continue

                ymin, xmin, ymax, xmax = boxes[i]
                xmin = int(xmin * width)
                xmax = int(xmax * width)
                ymin = int(ymin * height)
                ymax = int(ymax * height)

                detected_objects.append({
                    "class_name": class_name,
                    "score": float(scores[i]),
                    "box": [xmin, ymin, xmax, ymax],
                    "color": get_color_for_class(class_name)
                })

        # Рисуем bounding boxes на ОРИГИНАЛЬНОМ BGR изображении
        for obj in detected_objects:
            xmin, ymin, xmax, ymax = obj["box"]
            color = obj["color"]  # Цвета уже в BGR формате

            # Рисуем прямоугольник
            cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), color, 3)

            # Добавляем текст
            label = f"{obj['class_name']}: {obj['score']:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Рисуем подложку для текста
            cv2.rectangle(original_image,
                          (xmin, ymin - text_height - 10),
                          (xmin + text_width, ymin),
                          color, -1)

            # Рисуем текст (белый цвет в BGR)
            cv2.putText(original_image, label, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Сохраняем изображения
        uploads_dir = "static/uploads"
        os.makedirs(uploads_dir, exist_ok=True)

        # Сохраняем обработанное изображение (BGR)
        output_filename = f"detected_{file.filename}"
        output_path = f"{uploads_dir}/{output_filename}"
        cv2.imwrite(output_path, original_image)

        # Сохраняем оригинальное изображение (BGR)
        original_filename = f"original_{file.filename}"
        original_path = f"{uploads_dir}/{original_filename}"
        cv2.imwrite(original_path, image)  # image тоже в BGR

        # Статистика
        class_stats = {}
        for obj in detected_objects:
            class_name = obj["class_name"]
            class_stats[class_name] = class_stats.get(class_name, 0) + 1

        detection_history.append({
            'objects': detected_objects,
            'total_detected': len(detected_objects),
        })

        reports_dir = "static/reports"
        os.makedirs(reports_dir, exist_ok=True)
        report_filename = f"report_{len(detected_objects)}.json"
        report_data = {
            "detected_objects": detected_objects,
            "class_distribution": class_stats
        }

        with open(f"static/reports/{report_filename}", 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return templates.TemplateResponse("ssd_result.html", {
            "request": request,
            "detected_objects": detected_objects,
            "class_stats": class_stats,
            "image_url": f"/static/uploads/{output_filename}",
            "original_image_url": f"/static/uploads/{original_filename}",
            "total_detected": len(detected_objects),
            "used_threshold": threshold,
            "used_classes": ", ".join(valid_classes) if valid_classes else "все классы",
            "settings": {
                "threshold": threshold,
                "selected_classes": selected_classes
            },
            "report_url": f"/static/reports/{report_filename}"
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": str(e), "status_code": 500})

@router.get("/stats", response_class=HTMLResponse)
async def show_stats(request: Request):
    total_objects = 0
    class_distribution = {}

    for detection in detection_history:
        total_objects += detection['total_detected']
        for obj in detection['objects']:
            class_name = obj['class_name']
            if class_name in class_distribution:
                class_distribution[class_name] += 1
            else:
                class_distribution[class_name] = 1

    avg_obj = total_objects / 2

    top_objects = []
    for class_name, count in class_distribution.items():
        top_objects.append((class_name, count))
    top_objects.sort(key=lambda x: x[1], reverse=True)
    top_objects = top_objects[:5]
    detection_history.clear()

    return templates.TemplateResponse("ssd_stats.html", {
        "request": request,
        'total_objects': total_objects,
        'avg_obj': avg_obj,
        'top_objects': top_objects,
        'class_distribution': class_distribution
    })
