# Converti un mdoel au format yolo (.pt) en un model au format CreateML (.mlmodel)
# Steps:
# Installer les dépendances nécessaires
# !pip install coremltools onnx onnx-simplifier ultralytics
# Étape 2 : Exporter le modèle YOLOv8 en ONNX

from ultralytics import YOLO

# Charger le modèle YOLOv8 entraîné
model = YOLO("path/to/best.pt")  # Remplace par le chemin de ton modèle YOLOv8 entraîné

# Exporter le modèle au format ONNX
model.export(format="onnx", imgsz=640)  # Taille de l'image (imgsz) doit correspondre à la taille utilisée lors de l'entraînement

