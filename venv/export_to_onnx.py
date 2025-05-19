from ultralytics import YOLO

# Cargar el modelo entrenado (ajusta la ruta si es necesario)
model = YOLO("X:/PycharmProjects/Fruit-and-Vegetable-Classifier/runs/detect/train3/weights/best.pt")

# Exportar a ONNX
model.export(format="onnx", opset=12, simplify=True)


