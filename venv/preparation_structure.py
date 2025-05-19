import os
import shutil
import random

# Ruta base donde se encuentran las imágenes organizadas en subdirectorios
dataset_path = 'X:/Fruit And Vegetable Diseases Dataset'
images_base_path = os.path.join(dataset_path, 'images')

# Directorios donde se guardarán las imágenes de entrenamiento y validación
train_dir = os.path.join(images_base_path, 'train')
val_dir = os.path.join(images_base_path, 'val')

# Crear las carpetas de entrenamiento y validación si no existen
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Establecer una semilla aleatoria para reproducibilidad
random.seed(42)

# Recorrer todas las imágenes en 'images' y sus subdirectorios (sin incluir la carpeta "images")
all_images = []
for root, dirs, files in os.walk(images_base_path):
    # Solo procesamos los archivos que están dentro de las subcarpetas de imágenes
    if root != images_base_path:  # Evitar la carpeta raíz 'images'
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Verificar extensiones de imagen
                all_images.append(os.path.join(root, file))  # Agregar la ruta completa de la imagen

# Dividir el conjunto de imágenes en entrenamiento (80%) y validación (20%)
train_images = random.sample(all_images, int(0.8 * len(all_images)))
val_images = [img for img in all_images if img not in train_images]

# Función para mover las imágenes a las carpetas correspondientes
def move_images(images, dest_dir):
    for img in images:
        # Crear subcarpetas en el directorio destino para imitar la estructura original
        relative_path = os.path.relpath(img, images_base_path)  # Obtener la ruta relativa de la imagen
        dest_path = os.path.join(dest_dir, relative_path)  # Ruta destino en la carpeta de entrenamiento o validación

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # Crear los directorios necesarios
        shutil.move(img, dest_path)  # Mover la imagen a la carpeta correspondiente

# Mover las imágenes de entrenamiento y validación
move_images(train_images, train_dir)
move_images(val_images, val_dir)

print(f"Total imágenes de entrenamiento: {len(train_images)}")
print(f"Total imágenes de validación: {len(val_images)}")

# Crear archivo de configuración para YOLO
config_path = os.path.join(dataset_path, 'dataset.yaml')
with open(config_path, 'w') as f:
    f.write(f"path: {dataset_path}\n")  # Ruta principal del dataset
    f.write(f"train: images/train/\n")  # Carpeta de imágenes de entrenamiento
    f.write(f"val: images/val/\n")  # Carpeta de imágenes de validación
    f.write("\n")
    f.write("names:\n")
    for i, name in enumerate([
        "Apple_Healthy", "Apple_Rotten", "Banana_Healthy", "Banana_Rotten",
        "Bellpepper_Healthy", "Bellpepper_Rotten", "Carrot_Healthy", "Carrot_Rotten",
        "Cucumber_Healthy", "Cucumber_Rotten", "Grape_Healthy", "Grape_Rotten",
        "Guava_Healthy", "Guava_Rotten", "Jujube_Healthy", "Jujube_Rotten",
        "Mango_Healthy", "Mango_Rotten", "Orange_Healthy", "Orange_Rotten",
        "Pomegranate_Healthy", "Pomegranate_Rotten", "Potato_Healthy", "Potato_Rotten",
        "Strawberry_Healthy", "Strawberry_Rotten", "Tomato_Healthy", "Tomato_Rotten"
    ]):
        f.write(f"  {i}: {name}\n")  # Escribir las clases en el archivo .yaml

print("División completa y archivo dataset.yaml generado.")
