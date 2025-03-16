import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import matplotlib.pyplot as plt

# Configuración del dataset
data_path = 'X:/Fruit And Vegetable Diseases Dataset'
data_path = pathlib.Path(data_path)

# Parámetros
time_size = (150, 150)
batch_size = 64  # Aumento del batch size para mejorar estabilidad

# Carga y preprocesamiento del dataset con aumento de datos
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,  # Normalización
    validation_split=0.2,  # División en entrenamiento y validación
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_ds = datagen.flow_from_directory(
    data_path,
    target_size=time_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_ds = datagen.flow_from_directory(
    data_path,
    target_size=time_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Modelo CNN mejorado
model = keras.Sequential([
    # Primera capa convolucional
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Segunda capa convolucional
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Tercera capa convolucional
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Cuarta capa convolucional
    layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Aplanado y capas totalmente conectadas
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(28, activation='softmax')  # 28 clases
])

# Compilación
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25  # Aumentado para mejor aprendizaje
)

# Guardar el modelo
model.save("fruit_veggie_classifier-V2.h5")

# Graficar precisión y pérdida
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(history.history['loss'])])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
