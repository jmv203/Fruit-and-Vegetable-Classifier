import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import pathlib

# Cargar el modelo entrenado
model = tf.keras.models.load_model("fruit_veggie_classifier.h5")


# Función para predecir una imagen
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Redimensionar la imagen
    img_array = image.img_to_array(img)  # Convertir a array
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch
    img_array /= 255.0  # Normalización

    prediction = model.predict(img_array)[0][0]  # Obtener la predicción
    class_label = "Buen estado" if prediction < 0.5 else "Mal estado"

    # Mostrar la imagen y el resultado
    plt.imshow(img)
    plt.title(f"Predicción: {class_label}")
    plt.axis("off")
    plt.show()

    return class_label


# Prueba con una imagen de ejemplo
img_path = "X:/Fruit And Vegetable Diseases Dataset/rottenCarrot (53).jpg"  # Ruta de prueba
result = predict_image(img_path)
print(f"La imagen se clasifica como: {result}")
