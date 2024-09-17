# Convert a CreateML dataset to YOLO format
# 

import json
import os
from PIL import Image

# Fonction pour convertir les coordonnées CreateML en format YOLO
def convert_to_yolo_format(img_width, img_height, box):
    # Calcul des coordonnées YOLO : (x_center, y_center, width, height) normalisés
    x_center = (box["x"] + box["width"] / 2) / img_width
    y_center = (box["y"] + box["height"] / 2) / img_height
    width = box["width"] / img_width
    height = box["height"] / img_height
    return x_center, y_center, width, height

# Chemins des répertoires
createml_annotations_dir = 'path/to/createml/annotations'  # Répertoire contenant les fichiers JSON CreateML
createml_images_dir = 'path/to/createml/images'            # Répertoire contenant les images
yolo_output_dir = 'path/to/yolo/output'                    # Répertoire de sortie pour YOLO annotations

# Crée le répertoire pour les annotations YOLO s'il n'existe pas
os.makedirs(yolo_output_dir, exist_ok=True)

# Parcourir tous les fichiers JSON dans le répertoire d'annotations CreateML
for annotation_file in os.listdir(createml_annotations_dir):
    if annotation_file.endswith('.json'):
        # Lire le fichier JSON CreateML
        with open(os.path.join(createml_annotations_dir, annotation_file), 'r') as f:
            data = json.load(f)

        # Obtenir le nom de l'image associée
        image_filename = annotation_file.replace('.json', '.jpg')  # Assumes images are in .jpg
        image_path = os.path.join(createml_images_dir, image_filename)

        # Ouvrir l'image pour obtenir ses dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Fichier d'annotations YOLO à créer
        yolo_annotation_file = os.path.join(yolo_output_dir, annotation_file.replace('.json', '.txt'))

        # Ouvrir le fichier d'annotations YOLO en écriture
        with open(yolo_annotation_file, 'w') as yolo_f:
            # Parcourir les annotations d'objets dans le fichier CreateML
            for obj in data:
                label = obj["label"]  # Classe de l'objet
                box = obj["coordinates"]  # Coordonnées de la boîte

                # Convertir les coordonnées CreateML en format YOLO
                x_center, y_center, width, height = convert_to_yolo_format(img_width, img_height, box)

                # Si tu as un mapping de classes, tu peux l'utiliser ici, sinon utiliser un index
                class_id = 0  # Modifier selon la classe, ou créer un mapping label -> class_id

                # Écrire dans le fichier d'annotations YOLO (class_id, x_center, y_center, width, height)
                yolo_f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"Converted {annotation_file} to {yolo_annotation_file}")

print("Conversion terminée.")
