import os
import json
import cv2
import numpy as np
from ultralytics.data.converter import convert_coco

#Directorios
dataset_dir = "X:/Fruit And Vegetable Diseases Datasetcp/images/"
output_dir = "X:/Fruit And Vegetable Diseases Datasetcp/processed_images_segmentadas"

os.makedirs(output_dir, exist_ok=True)

#Definir categorías
categories = [
    "Apple__Healthy", "Apple__Rotten", "Banana__Healthy", "Banana__Rotten",
    "Bellpepper__Healthy", "Bellpepper__Rotten", "Carrot__Healthy", "Carrot__Rotten",
    "Cucumber__Healthy", "Cucumber__Rotten", "Grape__Healthy", "Grape__Rotten",
    "Guava__Healthy", "Guava__Rotten", "Jujube__Healthy", "Jujube__Rotten",
    "Mango__Healthy", "Mango__Rotten", "Orange__Healthy", "Orange__Rotten",
    "Pomegranate__Healthy", "Pomegranate__Rotten", "Potato__Healthy", "Potato__Rotten",
    "Strawberry__Healthy", "Strawberry__Rotten", "Tomato__Healthy", "Tomato__Rotten"
]

# Crear mapeo de categorías
category_id_map = {name: idx + 1 for idx, name in enumerate(categories)}

# Estructura COCO
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": i + 1, "name": name} for i, name in enumerate(categories)]
}
annotation_id = 1
img_idx = 1  # Iniciar con id de imagen como 1


# Función para procesar imágenes manteniendo la estructura de directorios
def process_images_from_dir(subdir, subset_name):
    global img_idx, annotation_id

    # Recorremos las categorías dentro de la carpeta 'train' o 'val'
    for category in os.listdir(subdir):
        category_path = os.path.join(subdir, category)
        if not os.path.isdir(category_path):
            continue

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Convertir a escala de grises y aplicar desenfoque
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Aplicar segmentación adaptativa
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Eliminar pequeños ruidos
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Encontrar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Mantener la estructura del directorio en la salida
            output_category_dir = os.path.join(output_dir, subset_name, category)
            os.makedirs(output_category_dir, exist_ok=True)
            processed_img_path = os.path.join(output_category_dir, img_name)

            # Agregar imagen a la estructura COCO con ruta relativa
            relative_img_path = os.path.relpath(processed_img_path, output_dir)
            coco_data["images"].append({
                "id": img_idx,
                "file_name": relative_img_path,  # Guardar ruta relativa
                "height": img.shape[0],
                "width": img.shape[1]
            })

            # Guardar anotaciones de cada contorno detectado
            for contour in contours:
                if cv2.contourArea(contour) < 1000:  # Filtrar ruidos pequeños
                    continue

                # Obtener rectángulo delimitador
                x, y, w, h = cv2.boundingRect(contour)
                segmentation = contour.flatten().tolist()

                # Crear anotación para el contorno
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_idx,
                    "category_id": category_id_map[category],  # Asignar ID de la categoría
                    "bbox": [x, y, w, h],
                    "segmentation": [segmentation],
                    "area": cv2.contourArea(contour),
                    "iscrowd": 0
                })
                annotation_id += 1

                # Dibujar bounding box en la imagen
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, category, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Guardar la imagen procesada con anotaciones en su carpeta correspondiente
            cv2.imwrite(processed_img_path, img)

            # Incrementar el índice de la imagen
            img_idx += 1


# Procesar imágenes de 'train' y 'val'
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")

process_images_from_dir(train_dir, "train")
process_images_from_dir(val_dir, "val")

# Guardar las anotaciones COCO en formato JSON
json_output_path = "X:/Fruit And Vegetable Diseases Datasetcp/annotations/dataset_coco.json"
with open(json_output_path, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f" Anotaciones guardadas en {json_output_path}")



