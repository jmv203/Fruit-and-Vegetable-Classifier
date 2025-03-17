import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from image_prediction import predict_image  # Importamos la función de predicción del image_prediction.py


def process_rotten_image(img_path):
    # Predecir la fruta y su estado
    fruit, condition = predict_image(img_path)

    if condition == "Rotten":
        print(f"Aplicando procesamiento de imagen a {fruit} podrida...")

    #Funciones de tratamiento...

    else:
        print(f"La fruta {fruit} es saludable. No se aplica procesamiento.")


# Prueba con una imagen nueva
img_path = "X:/new/download.jpg"  # Ruta de prueba
process_rotten_image(img_path)
