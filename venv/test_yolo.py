from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from collections import Counter

# 1. Cargar el modelo entrenado
model = YOLO(
    "X:/PycharmProjects/Fruit-and-Vegetable-Classifier/runs/detect/train12/weights/best.torchscript.pt")  # Ruta a tu modelo entrenado

# 2. Probar con una imagen
image_path = "X:/Fruit And Vegetable Diseases Dataset/images/train/Apple__Healthy/freshApple (1-1).png"
results = model.predict(image_path, conf=0.5)  # Umbral de confianza más bajo (0.3)

# 3. Visualizar resultados
for result in results:
    img_with_boxes = result.plot()
    plt.imshow(img_with_boxes[:, :, ::-1])
    plt.axis("off")
    plt.show()

    # 4. Contar las detecciones por clase
    detections = []  # Lista para almacenar las clases detectadas
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
        detections.append(class_name)
        print(f"- {class_name} (Confianza: {confidence:.2f})")

    # 5. Contar la cantidad de cada clase detectada
    detection_count = Counter(detections)
    print("\nCantidad de cada clase detectada:")
    for class_name, count in detection_count.items():
        print(f"{class_name}: {count}")
