from deep_sort import DeepSort
from yolo import YOLO

# Initialisation des modèles
yolo = YOLO(model_path="path_to_yolo_weights")
deepsort = DeepSort()

# Boucle de traitement des images (ou frames d'une vidéo)
for frame in video_frames:
    # Détection des objets avec YOLO
    detections = yolo.detect(frame)
    
    # Extraction des bounding boxes, scores et classes
    bboxes = [d["bbox"] for d in detections]
    scores = [d["score"] for d in detections]
    classes = [d["class"] for d in detections]
    
    # Suivi des objets avec Deep SORT
    tracked_objects = deepsort.update(bboxes, scores, classes, frame)
    
    # Affichage des IDs et bounding boxes
    for obj in tracked_objects:
        id = obj.track_id
        bbox = obj.bbox
        classe = obj.class_id
        print(f"Objet {classe} avec ID unique {id} détecté à {bbox}")