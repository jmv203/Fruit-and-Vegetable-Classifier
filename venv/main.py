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
batch_size = 32

# Carga y preprocesamiento del dataset
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # Normalización
    validation_split=0.2,  # División en entrenamiento y validación
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
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

# Modelo CNN
model = keras.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(28, activation='softmax')  # 28 clases
])

# Compilación
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Ajustado para múltiples clases
              metrics=['accuracy'])

# Entrenamiento
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Guardar el modelo
model.save("fruit_veggie_classifier.h5")

# Graficar precisión
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()