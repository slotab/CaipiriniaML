import cv2
import os


def visualize_yolo_annotation(image_path, annotation_path, class_names=None):
    # Charger l'image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Lire l'annotation YOLO
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    # Parcourir chaque annotation
    for line in lines:
        # Extraire les valeurs de l'annotation YOLO (classe, x_center, y_center, width, height)
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        bbox_width = float(parts[3])
        bbox_height = float(parts[4])

        # # Convertir les coordonnées relatives en coordonnées absolues
        x_center_abs = int(x_center * img_width)
        y_center_abs = int(y_center * img_height)
        bbox_width_abs = int(bbox_width * img_width)
        bbox_height_abs = int(bbox_height * img_height)

        # # Calculer les coordonnées du rectangle (en haut à gauche, en bas à droite)
        x_min = int(x_center_abs - bbox_width_abs / 2)
        y_min = int(y_center_abs - bbox_height_abs / 2)
        x_max = int(x_center_abs + bbox_width_abs / 2)   # Dessiner le rectangle de la bounding box sur l'image
        y_max = int(y_center_abs + bbox_height_abs / 2)

        color = (0, 255, 0)  # Vert
        thickness = 2
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        # Ajouter le nom de la classe (si fourni)
        if class_names:
            class_name = class_names[class_id]
            cv2.putText(img, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Afficher l'image annotée
    cv2.imshow('YOLO Annotation Visualization', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Exemple d'utilisation
image_path = '../../out/yolo/datasets/valid/images/aperol_002_0016_contrast_2.jpg'
annotation_path = '../../out/yolo/datasets/valid/labels/aperol_002_0016_contrast_2.txt'

class_names = ['martini',
               'cocacola',
               'gin',
               'vodka',
               'cointreau',
               'tonic',
               'rum',
               'curacao',
               'campari',
               'aperol']

visualize_yolo_annotation(image_path, annotation_path, class_names)
