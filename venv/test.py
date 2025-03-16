import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import pathlib

# Cargar el modelo entrenado
model = tf.keras.models.load_model("fruit_veggie_classifier-V2.h5")

# Lista de clases (ajústala según el orden en que se entrenó el modelo)
class_labels = [
    "Apple__Healthy", "Apple__Rotten", "Banana__Healthy", "Banana__Rotten",
    "Bellpepper__Healthy", "Bellpepper__Rotten", "Carrot__Healthy", "Carrot__Rotten",
    "Cucumber__Healthy", "Cucumber__Rotten", "Grape__Healthy", "Grape__Rotten",
    "Guava__Healthy", "Guava__Rotten", "Jujube__Healthy", "Jujube__Rotten",
    "Mango__Healthy", "Mango__Rotten", "Orange__Healthy", "Orange__Rotten",
    "Pomegranate__Healthy", "Pomegranate__Rotten", "Potato__Healthy", "Potato__Rotten",
    "Strawberry__Healthy", "Strawberry__Rotten", "Tomato__Healthy", "Tomato__Rotten"
]

# Función para predecir una imagen
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Redimensionar la imagen
    img_array = image.img_to_array(img)  # Convertir a array
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch
    img_array /= 255.0  # Normalización

    # Obtener predicción
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)  # Obtener el índice de la clase con mayor probabilidad
    predicted_label = class_labels[predicted_index]  # Obtener el nombre de la clase

    # Dividir en fruta y estado
    fruit_name, condition = predicted_label.split("__")

    # Mostrar la imagen y el resultado
    plt.imshow(img)
    plt.title(f"Fruta: {fruit_name}\nEstado: {condition}")
    plt.axis("off")
    plt.show()

    return fruit_name, condition

# Prueba con una imagen de ejemplo
img_path = "X:/Fruit And Vegetable Diseases Dataset/new/platano.jpg"  # Ruta de prueba
fruit, condition = predict_image(img_path)
print(f"Fruta detectada: {fruit}")
print(f"Estado: {condition}")
