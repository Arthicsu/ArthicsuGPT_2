import tensorflow as tf
from tensorflow import keras
from keras import layers

# Настройки
data_dir = "static/assets/satellite_photos"
img_height = 224
img_width = 224
batch_size = 32
epochs = 15

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
num_classes = len(class_names)

# УЛУЧШЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ДАННЫХ
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Загрузка предобученной модели VGG16
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

base_model.trainable = True

# Замораживаем всё, кроме последних 5 слоев
for layer in base_model.layers[:-5]:
    layer.trainable = False

# ДИАГНОСТИКА: посчитаем сколько слоев обучается
trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
total_count = len(base_model.layers)
print(f"Обучается {trainable_count}/{total_count} слоев VGG16")

# Создание модели с переносом обучения
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# УЛУЧШЕННАЯ КОМПИЛЯЦИЯ
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
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
model.save('src/models/satellite_model_VGG16_transfer.h5')
print("Модель с переносом обучения сохранена как 'satellite_model_VGG16_transfer.h5'")