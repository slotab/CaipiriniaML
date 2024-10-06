from ultralytics import YOLO

#model = YOLO("yolov8n.pt")  # Pour commencer, on utilise le modèle YOLOv8 pré-entraîné
model = YOLO("yolov8n.yaml") # form scratch

# Définir le chemin vers le dataset (assurez-vous que le dataset est correctement structuré)
data_yaml = "dataset/data.yaml"  # Le fichier .yaml qui décrit le dataset

# Entraîner le modèle
model.train(data=data_yaml, epochs=25, imgsz=640, name="lime", device="mps") # mps = Apple M2
