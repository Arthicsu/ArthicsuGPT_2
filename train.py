import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import json, os


data_dir = "/content/drive/MyDrive/satellite_photos"
img_height = 224
img_width = 224
batch_size = 32
epochs = 15

# ПОДГОТОВКА ДАННЫХ
# В PyTorch: используем ImageFolder + трансформации
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),  # Изменение размера как в TensorFlow
    transforms.ToTensor(),  # Конвертация в тензор PyTorch
    # Нормализация - аналог rescaling в TensorFlow
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка датасета - аналог image_dataset_from_directory в TensorFlow
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Разделение на train/val - аналог validation_split=0.2 в TensorFlow
train_size = int(0.8 * len(dataset))  # 80% для обучения
val_size = len(dataset) - train_size  # 20% для валидации
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Создание DataLoader - аналог batch_size в TensorFlow
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # shuffle как в TensorFlow
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Получаем названия классов - полностью идентично TensorFlow
class_names = dataset.classes
print("Классы:", class_names)

# ВИЗУАЛИЗАЦИЯ - концепт такой же как в TensorFlow
plt.figure(figsize=(10, 10))
for images, labels in train_loader:  # Аналог train_ds.take(1) в TF
    for i in range(min(9, len(images))):  # Показываем до 9 изображений
        ax = plt.subplot(3, 3, i + 1)
        # В PyTorch изображения в формате (C, H, W), нужно преобразовать для matplotlib
        img = images[i].permute(1, 2, 0).numpy()  # Меняем порядок осей: CHW -> HWC
        # Денормализация для корректного отображения
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img.clip(0, 1)  # Обрезаем значения до [0, 1]
        plt.imshow(img)
        plt.title(class_names[labels[i]])  # Название класса как в TF
        plt.axis("off")
    break  # Берем только первый батч, аналог .take(1) в TF
plt.show()

# СОЗДАНИЕ МОДЕЛИ - слои аналогичны TensorFlow версии
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
    nn.Linear(256, len(class_names))  # Выходной слой: 128 -> количество классов
)

# КОМПИЛЯЦИЯ МОДЕЛИ - аналог model.compile() в TensorFlow
criterion = nn.CrossEntropyLoss()  # Функция потерь - аналог sparse_categorical_crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Оптимизатор Adam как в TF

# Вывод структуры модели - аналог model.summary() в TF
print("Структура модели:")
print(model)

# ОБУЧЕНИЕ МОДЕЛИ - концепт эпох и батчей такой же как в TF
train_losses = []  # Для сохранения истории потерь
val_accuracies = []  # Для сохранения истории точности

for epoch in range(epochs):  # Цикл по эпохам как в TF
    # РЕЖИМ ОБУЧЕНИЯ
    model.train()  # Устанавливаем модель в режим обучения
    running_loss = 0.0  # Для накопления потерь за эпоху

    for images, labels in train_loader:  # Итерация по батчам как в TF
        # ОТЛИЧИЕ OT TENSORFLOW: нужно обнулять градиенты
        optimizer.zero_grad()  # Обнуляем градиенты от предыдущего батча

        # ПРЯМОЙ ПРОХОД - аналог model.predict() в TF но с вычислением градиентов
        outputs = model(images)  # Пропускаем данные через модель
        loss = criterion(outputs, labels)  # Вычисляем функцию потерь

        # ОБРАТНЫЙ ПРОХОД - аналог автоматического в TF но явный в PyTorch
        loss.backward()  # Вычисляем градиенты (обратное распространение)
        optimizer.step()  # Обновляем веса модели

        running_loss += loss.item()  # Суммируем потери

    # ВАЛИДАЦИЯ - концепт такой же как в TF
    model.eval()  # Режим оценки (отключаем Dropout и т.д.)
    correct = 0  # Счетчик правильных предсказаний
    total = 0  # Общее количество примеров

    with torch.no_grad():  # Отключаем вычисление градиентов для экономии памяти
        for images, labels in val_loader:
            outputs = model(images)  # Прямой проход
            _, predicted = torch.max(outputs, 1)  # Берем класс с максимальной вероятностью
            total += labels.size(0)  # Увеличиваем счетчик общего количества
            correct += (predicted == labels).sum().item()  # Считаем правильные предсказания

    # ВЫЧИСЛЯЕМ МЕТРИКИ
    epoch_loss = running_loss / len(train_loader)  # Средняя потеря за эпоху
    accuracy = 100 * correct / total  # Точность в процентах

    train_losses.append(epoch_loss)  # Сохраняем потери для графиков
    val_accuracies.append(accuracy)  # Сохраняем точность для графиков

    # Вывод прогресса - аналогично TF
    print(f'Эпоха [{epoch + 1}/{epochs}], Потери: {epoch_loss:.4f}, Точность: {accuracy:.2f}%')

# СОХРАНЕНИЕ МОДЕЛИ - аналог model.save() в TF
torch.save(model.state_dict(), 'satellite_model_pytorch.pth')  # Сохраняем веса модели
print("Модель сохранена как 'satellite_model_pytorch.pth'")

# СОХРАНЕНИЕ КЛАССОВ - полностью идентично TF версии
with open('satellite_classes_pytorch.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False)
print("Названия классов сохранены в 'satellite_classes_pytorch.json'")

# ВИЗУАЛИЗАЦИЯ ОБУЧЕНИЯ - аналогично TF но с использованием matplotlib
plt.figure(figsize=(12, 4))

# График потерь
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Потери при обучении')  # Аналог 'Training Loss'
plt.xlabel('Эпоха')
plt.ylabel('Потери')

# График точности
plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title('Точность на валидации')  # Аналог 'Validation Accuracy'
plt.xlabel('Эпоха')
plt.ylabel('Точность (%)')

plt.tight_layout()  # Улучшаем расположение графиков
plt.show()