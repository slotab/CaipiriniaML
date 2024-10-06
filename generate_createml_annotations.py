import os
import json
from PIL import Image

HOME = os.getcwd()
print(HOME)


def load_class_map(labels_file_path):
    """
    Lit un fichier de labels de classes (un label par ligne) et génère un dictionnaire class_map.
    :param labels_file_path: Chemin vers le fichier contenant les labels de classes.
    :return: Dictionnaire class_map qui mappe chaque indice à un nom de classe.
    """
    class_map = {}
    
    with open(labels_file_path, 'r') as file:
        for idx, line in enumerate(file.readlines()):
            class_label = line.strip()  # Enlever les espaces et sauts de ligne
            class_map[idx] = class_label  # Associer l'indice à la classe

    return class_map


def yolo_to_createml(class_map, image_path, yolo_annotation_path):
    image_filename = os.path.basename(image_path)
    
    # Charger l'image pour obtenir ses dimensions
    with Image.open(image_path) as img:
        image_width, image_height = img.size
            
    # Lire l'annotation YOLO    
    with open(yolo_annotation_path, 'r') as f:
        yolo_annotations = f.readlines()
        
    annotations = []

    # Parcourir chaque ligne d'annotation YOLO
    for annotation in yolo_annotations:
        try:
            class_id, x_center, y_center, width, height = map(float, annotation.strip().split())

            # Convertir les coordonnées YOLO en coordonnées CreateML
            x_top_left = (x_center) * image_width
            y_top_left = (y_center) * image_height
            width_bbox = width * image_width
            height_bbox = height * image_height

            # Ajouter l'annotation dans le format CreateML
            annotations.append({
                "label": class_map[int(class_id)],
                "coordinates": {
                    "x": x_top_left,
                    "y": y_top_left,
                    "width": width_bbox,
                    "height": height_bbox
                }
            })
        except ValueError as e:
            # Afficher une alerte pour la ligne incorrecte et continuer avec les autres annotations
            print(f"Erreur lors du traitement de l'annotation pour l'image '{yolo_annotation_path}': {e}")
            continue
    
    return {
        "image": image_filename,
        "annotations": annotations
    }
        
        
def process(class_map, basedir):
    image_folder = f"{basedir}/images"
    yolo_annotations_folder = f"{basedir}/labels"
    output_json_file = f"{basedir}/images/_annotations.createml.json"
    
    all_annotations = []

    # Parcourir les annotations YOLO et générer les fichiers JSON CreateML
    for yolo_annotation_filename in os.listdir(yolo_annotations_folder):
        if yolo_annotation_filename.endswith('.txt'):
            image_filename = f"{os.path.splitext(yolo_annotation_filename)[0]}.jpg"  # Ou '.png' selon ton dataset
            image_path = os.path.join(image_folder, image_filename)
            yolo_annotation_path = os.path.join(yolo_annotations_folder, yolo_annotation_filename)
            annotation_data = yolo_to_createml(class_map, image_path, yolo_annotation_path)
            
            all_annotations.append(annotation_data)
            
    # Sauvegarder toutes les annotations dans un seul fichier JSON
    with open(output_json_file, 'w') as json_file:
        json.dump(all_annotations, json_file, indent=4)

    print(f"Fichier JSON CreateML généré: {output_json_file}")


# Mapping des class_id aux noms des classes
class_map = load_class_map("caipirinia_labels.txt")

process(class_map, f"{HOME}/dataset/train")
process(class_map, f"{HOME}/dataset/valid")
process(class_map, f"{HOME}/dataset/test")

