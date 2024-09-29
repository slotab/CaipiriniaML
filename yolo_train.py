from ultralytics import YOLO

# Charger le modèle YOLOv8 pré-entrainé (ici, YOLOv8n qui est une version plus légère)
model = YOLO("yolov8n.pt")  # Pour commencer, on utilise le modèle YOLOv8 pré-entraîné
#model = YOLO("yolov8n.yaml") # form scratch

# Définir le chemin vers le dataset (assurez-vous que le dataset est correctement structuré)
data_yaml = "caipirinia.yaml"  # Le fichier .yaml qui décrit le dataset

# Entraîner le modèle
model.train(data=data_yaml, epochs=10, imgsz=640, batch=16, name="yolov8_caipirinia")

