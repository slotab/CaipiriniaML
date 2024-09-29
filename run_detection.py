from ultralytics import YOLO
import cv2


def load_model(model_path):
    """
    Charge le modèle YOLO à partir d'un fichier.
    """
    model = YOLO(model_path, task="detect")
    return model


def infer_and_draw(image_path, model):
    """
    Effectue une inférence sur une image, dessine les boîtes de détection et affiche l'image annotée.
    """
    # Charger l'image
    image = cv2.imread(image_path)

    # Obtenir les dimensions de l'image
    height, width, _ = image.shape

    # Effectuer l'inférence avec le modèle YOLO
    results = model(image_path)

    # Parcourir les résultats de détection
    for result in results:
        for box in result.boxes:
            # Extraire les coordonnées de la boîte (format xywh)
            x, y, w, h = box.xywh[0]

            # Convertir en format (xmin, ymin, xmax, ymax)
            xmin = int(x - w / 2)
            ymin = int(y - h / 2)
            xmax = int(x + w / 2)
            ymax = int(y + h / 2)

            # Extraire le nom de la classe et la confiance
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = box.conf[0]

            # Dessiner la boîte de détection
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)

            # Ajouter le texte (classe et confiance)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    # Afficher l'image annotée
    cv2.imshow("YOLO Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Exemple d'utilisation
if __name__ == "__main__":

    model_path = "/Users/bslota/IdeaProjects/summer-challenge-ia/runs/yolov8_caipirinia_model3/weights/best.mlpackage"  # Remplace par le chemin vers ton modèle YOLO
    image_path = "/Users/bslota/IdeaProjects/summer-challenge-ia/caipirinia/images/images_IMG_7010.jpg"  # Remplace par le chemin vers ton image

    # Charger le modèle
    model = load_model(model_path)

    # Effectuer l'inférence et afficher les détections
    infer_and_draw(image_path, model)
