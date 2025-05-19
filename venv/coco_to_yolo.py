import os
import json
import ijson  # Para procesamiento eficiente de JSON


#  Rutas de entrada y salida
coco_json_path = "X:/Fruit And Vegetable Diseases Dataset/annotations/dataset_coco.json"
output_yolo_path = "X:/Fruit And Vegetable Diseases Dataset/labels/"


#  Crear carpetas base si no existen
os.makedirs(os.path.join(output_yolo_path, "train"), exist_ok=True)
os.makedirs(os.path.join(output_yolo_path, "val"), exist_ok=True)

#  Diccionarios para almacenar información
category_map = {}  # COCO ID → Nombre de categoría
image_dict = {}  # Image ID → Ruta de la imagen
image_data = {}  # Image ID → {width, height}

#  Leer el JSON de COCO de manera eficiente
with open(coco_json_path, "r") as f:
    # Extraer categorías
    for category in ijson.items(f, "categories.item"):
        category_map[category["id"]] = category["name"]

    # Volver a abrir el archivo para imágenes (reset)
    f.seek(0)

    # Extraer imágenes y almacenarlas en un diccionario
    for image in ijson.items(f, "images.item"):
        image_dict[image["id"]] = image["file_name"].replace("\\", "/")
        image_data[image["id"]] = {"width": image["width"], "height": image["height"]}

    # Volver a abrir el archivo para anotaciones
    f.seek(0)

    # Procesar anotaciones
    for annotation in ijson.items(f, "annotations.item"):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"] - 1  # YOLO usa índices desde 0
        bbox = annotation["bbox"]

        #  Obtener ruta de la imagen correspondiente
        if image_id not in image_dict:
            print(f" Error: No se encontró la imagen con ID {image_id}. Se omite la anotación.")
            continue

        image_path = image_dict[image_id]

        #  Determinar si la imagen pertenece a train o val
        if "train" in image_path:
            subset = "train"
        elif "val" in image_path:
            subset = "val"
        else:
            print(f" Error: No se puede determinar train/val para {image_path}")
            continue

        #  Asegurar que el directorio de la categoría existe
        category_name = category_map[category_id + 1]  # Ajustar índice
        category_label_dir = os.path.join(output_yolo_path, subset, category_name)
        os.makedirs(category_label_dir, exist_ok=True)

        #  Generar nombre del archivo de etiqueta
        label_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(category_label_dir, label_filename)

        #  Obtener dimensiones de la imagen desde el diccionario
        img_width = image_data[image_id]["width"]
        img_height = image_data[image_id]["height"]

        #  Calcular coordenadas normalizadas para YOLO
        x_center = (bbox[0] + bbox[2] / 2) / img_width
        y_center = (bbox[1] + bbox[3] / 2) / img_height
        width = bbox[2] / img_width
        height = bbox[3] / img_height

        yolo_annotation = f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

        #  Guardar anotaciones en el archivo correspondiente
        with open(label_path, "a") as label_file:
            label_file.write(yolo_annotation)

print(" Conversión de COCO a YOLO completada.")

