from ultralytics import YOLO
import json
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == '__main__':
    # Verificar si CUDA está disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Entrenando en el dispositivo: {device}")

    # Definir el archivo de configuración del dataset
    yaml_path = "X:/Fruit And Vegetable Diseases Dataset/dataset.yaml"

    # Cargar el modelo YOLOv8 preentrenado
    model = YOLO("yolov8n.pt")

    # Entrenar el modelo
    model.train(data=yaml_path, epochs=50, imgsz=416, batch=8, workers=0, verbose=False, device=device)

    # Guardar el modelo entrenado
    model_path = "yolov8_fruits_model.pt"
    model.export(format="torchscript", imgsz=640, simplify=True)  # Guardar en formato TorchScript
    print(f"Modelo guardado en: {model_path}")

    # Cargar los resultados del entrenamiento
    results_path = 'runs/detect/train/results.json'  # Ruta del JSON con métricas
    if not os.path.exists(results_path):
        print(f"Error: No se encontró el archivo {results_path}. Asegúrate de que el entrenamiento finalizó correctamente.")
        exit()

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Extraer métricas del entrenamiento
    epochs = list(range(1, len(results['results']) + 1))
    train_loss = [x['train/loss'] for x in results['results']]
    val_loss = [x['val/loss'] for x in results['results']]
    mAP_50 = [x['metrics/mAP_0.5'] for x in results['results']]
    mAP_50_95 = [x['metrics/mAP_0.5_0.95'] for x in results['results']]
    precision = [x['metrics/precision'] for x in results['results']]
    recall = [x['metrics/recall'] for x in results['results']]

    # Graficar pérdidas de entrenamiento y validación
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graficar mAP@0.5 y mAP@0.5-0.95
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mAP_50, label="mAP@0.5")
    plt.plot(epochs, mAP_50_95, label="mAP@0.5-0.95")
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('mAP during training')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graficar precisión y recall
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, precision, label="Precision")
    plt.plot(epochs, recall, label="Recall")
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Precision and Recall during training')
    plt.legend()
    plt.grid(True)
    plt.show()
