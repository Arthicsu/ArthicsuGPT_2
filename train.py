import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Настройки
data_dir = "/content/drive/MyDrive/satellite_photos"
img_height = 180
img_width = 180
batch_size = 32
epochs = 4

# Создание датасетов
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Получаем названия классов
class_names = train_ds.class_names
print("Классы:", class_names)

# Визуализация нескольких изображений
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Настройка производительности
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Создание модели для цветов (более сложная, чем для MNIST)
model = keras.Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Обучение
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Сохранение модели
model.save('satellite_model.h5')
print("Модель сохранена как 'satellite_model.h5'")

# Сохранение названий классов
import json

with open('satellite_classes.json', 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False)
print("Названия классов сохранены в 'satellite_classes.json'")


