from ultralytics import YOLO
import os
import cv2

###
# Run alive detection on a video 
###

HOME = os.getcwd()
print(HOME)

# VIDEO TO DETECT
INPUT_VIDEO_PATH = f"{HOME}/../videos/tests/IMG_7034.MOV"
# MODEL TO USE
TRAINED_MODEL_PATH = f"{HOME}/../model/bottle/best.pt"

model = YOLO(TRAINED_MODEL_PATH)

# Run inference on the source
#results = model(INPUT_VIDEO_PATH, stream=True)

# Lire la vidéo
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)


while cap.isOpened():
    ret, frame = cap.read()  # Lire une frame de la vidéo
    if not ret:
        break

    # Exécuter la détection YOLO sur cette frame
    results = model(frame)

    # Extraire les informations des prédictions (boîtes englobantes, classes, etc.)
    for result in results:
        boxes = result.boxes  # Boîtes englobantes
        for box in boxes:
            # Extraire les coordonnées de la boîte
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()  # Confiance
            cls = int(box.cls[0])  # Classe de l'objet
            
            # Dessiner la boîte et l'étiquette sur la frame
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

    # Afficher la frame avec les prédictions
    cv2.imshow('YOLO Detection', frame)

    # Sortir en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
