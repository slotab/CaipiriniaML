import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Charger le modèle SSD MobileNetV2 depuis TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Charger les labels COCO (les noms des classes)
# Télécharger les labels de COCO ou les définir manuellement
LABELS_PATH = '../coco_labels.txt'  # Fichier contenant les noms des labels
with open(LABELS_PATH, 'r') as f:
    class_labels = f.read().splitlines()

# Fonction pour charger et prétraiter l'image
def load_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir BGR en RGB pour compatibilité avec le modèle
    return image_rgb

# Fonction pour effectuer la détection d'objets
def detect_objects(image):
    # Redimensionner l'image au format requis par le modèle
    input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]  # Ajouter une dimension pour le batch

    # Effectuer la détection
    detections = detector(input_tensor)

    # Extraire les boîtes, scores et classes
    bboxes = detections['detection_boxes'][0].numpy()  # Boîtes englobantes
    class_ids = detections['detection_classes'][0].numpy().astype(np.int32)  # Classes des objets
    scores = detections['detection_scores'][0].numpy()  # Scores de confiance
    return bboxes, class_ids, scores

# Fonction pour dessiner des cadres autour des objets détectés et afficher les labels
def draw_bounding_boxes(image, bboxes, class_ids, scores, threshold=0.5):
    h, w, _ = image.shape
    for i in range(len(scores)):
        if scores[i] >= threshold:
            # Extraire les coordonnées des boîtes englobantes
            ymin, xmin, ymax, xmax = bboxes[i]
            left, right, top, bottom = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)

            # Dessiner le cadre autour de l'objet
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Récupérer le nom de la classe
            class_label = class_labels[class_ids[i] - 1]  # Les IDs dans COCO commencent à 1, donc décalage de -1
            label = f"{class_label}: {scores[i]:.2f}"

            # Afficher le label au-dessus du cadre
            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    return image

# Fonction principale pour afficher les objets détectés
def detect_and_display(image_path):
    # Charger l'image
    image = load_image(image_path)

    # Détecter les objets dans l'image
    bboxes, class_ids, scores = detect_objects(image)

    # Dessiner les cadres autour des objets détectés et afficher les labels
    image_with_boxes = draw_bounding_boxes(image.copy(), bboxes, class_ids, scores)

    # Afficher l'image avec les cadres
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.show()

# Chemin de l'image à analyser
image_path = '../images/images_IMG_7010.jpg'

# Exécuter le pipeline de détection et d'affichage
detect_and_display(image_path)
