import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Configuración del dataset
data_path = 'X:/Fruit And Vegetable Diseases Dataset'
data_path = pathlib.Path(data_path)

# Parámetros
time_size = (150, 150)
batch_size = 64  # Aumentado para mejorar estabilidad del entrenamiento

# Aumento de datos mejorado
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # Normalización
    validation_split=0.2,  # División en entrenamiento y validación
    rotation_range=30,  # Aumentado para mejor generalización
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
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

# Modelo mejorado con Transfer Learning (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Congelar capas preentrenadas

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(28, activation='softmax')  # 28 clases
])

# Compilación del modelo
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20  # Aumentado para mejor convergencia
)

# Guardar el modelo
model.save("fruit_veggie_classifier_V3.h5")

# Graficar precisión y pérdida
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(loss)])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.show()