import cv2
import os
import shutil
import random
import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")

def resize_image(image_path, target_size=(640, 640)):
    # Ouvrir l'image
    with Image.open(image_path) as img:
        # Redimensionner l'image
        resized_img = img.resize(target_size)
        # Sauvegarder l'image redimensionnée
        resized_img.save(image_path)
        print(f"Image redimensionnée : {image_path}")

def resize_images_in_directory(root_directory, target_size=(640, 640)):
    # Parcourir récursivement tous les fichiers du répertoire
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            # Filtrer uniquement les fichiers d'image (jpg, png, etc.)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    resize_image(file_path, target_size)
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {file_path} : {e}")


import os
from PIL import Image, ImageEnhance
import random

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def augment_image(image_path, output_directory, base_filename):
    with Image.open(image_path) as img:
        # Générer des variations de luminosité
        for i in range(3):  # Par exemple, 3 variations de luminosité
            brightness_factor = random.uniform(0.5, 1.5)
            bright_image = adjust_brightness(img, brightness_factor)
            bright_image.save(os.path.join(output_directory, f"{base_filename}_brightness_{i}.jpg"))
            print(f"Image avec variation de luminosité générée : {base_filename}_brightness_{i}.jpg")

        # Générer des variations de contraste
        for i in range(3):  # Par exemple, 3 variations de contraste
            contrast_factor = random.uniform(0.5, 1.5)
            contrast_image = adjust_contrast(img, contrast_factor)
            contrast_image.save(os.path.join(output_directory, f"{base_filename}_contrast_{i}.jpg"))
            print(f"Image avec variation de contraste générée : {base_filename}_contrast_{i}.jpg")

def augment_images_in_directory(root_directory, output_directory):
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_directory, exist_ok=True)

    # Parcourir récursivement tous les fichiers du répertoire
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            # Filtrer uniquement les fichiers d'image (jpg, png, etc.)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                base_filename, ext = os.path.splitext(file)
                augment_image(file_path, output_directory, base_filename)




def split_frames(input_directory, output_directory, ratios):
    # Créer les sous-répertoires si nécessaire
    subdirs = ['train', 'valid', 'test']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_directory, subdir), exist_ok=True)

    # Lister tous les fichiers du répertoire d'entrée
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    # Mélanger les fichiers de manière aléatoire pour assurer une répartition aléatoire
    random.shuffle(files)

    # Calculer les indices de répartition
    total_files = len(files)
    train_cutoff = int(total_files * ratios[0])
    validation_cutoff = train_cutoff + int(total_files * ratios[1])

    # Répartition des fichiers dans les sous-répertoires
    for i, file in enumerate(files):
        src_path = os.path.join(input_directory, file)
        if i < train_cutoff:
            dst_path = os.path.join(output_directory, 'train', file)
        elif i < validation_cutoff:
            dst_path = os.path.join(output_directory, 'valid', file)
        else:
            dst_path = os.path.join(output_directory, 'test', file)

        shutil.move(src_path, dst_path)
        print(f"Déplacé : {src_path} -> {dst_path}")


# Function to extract frames from video
def extract_frames(video_path, output_dir, prefix, index, frame_interval=10):
    # Ouvrir le fichier vidéo
    cap = cv2.VideoCapture(video_path)

    # Obtenir les FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Nombre de frames par seconde : {fps}")

    # Vérifier si la vidéo a été ouverte correctement
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Index de la frame
    frame_index = 0
    saved_frame_count = 0

    while True:
        # Capturer image par image
        ret, frame = cap.read()

        # Si la frame a été lue correctement, traiter
        if ret:
            # Sauvegarder une frame toutes les `frame_interval` frames
            if frame_index % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f"{prefix}_{index:03d}_{saved_frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Saved {frame_filename}")
                saved_frame_count += 1
            frame_index += 1
        else:
            break  # Plus de frames à lire

    # Libérer l'objet de capture vidéo
    cap.release()
    print(f"Extraction terminée. Frames sauvegardées dans {output_dir}")


def extract_frames_through_directories(source_directory, destination_directory, frame_interval=10):
    # Créer le répertoire de destination s'il n'existe pas
    os.makedirs(destination_directory, exist_ok=True)

    index = 1

    # Parcourir récursivement le répertoire source
    for dirpath, dirnames, filenames in os.walk(source_directory):
        for filename in filenames:
            # Chemin complet du fichier source
            source_file_path = os.path.join(dirpath, filename)
            # folder name = tag/label
            label = dirpath.split('/')[-1]

            if source_file_path.endswith('.MOV'):
                extract_frames(source_file_path, destination_directory, label, index, frame_interval)
                print(f"Analysé : {source_file_path}")
                index+=1


# Fonction pour convertir une détection YOLOv8 en format Create ML
def yolo_to_createml_format(image_size, box, label=None):
    # x_min, y_min, x_max, y_max = box
    x, y, width, height = box
    img_width, img_height = image_size

    # Convertir les coordonnées en format Create ML
    # width = x_max - x_min
    # height = y_max - y_min

    return {
        "label": label,  # Ce champ sera mis à jour après la détection
        "coordinates": {
            "x": float(x),
            "y": float(y),
            "width": float(width),
            "height": float(height)
        }
    }

def compile_json_files(input_directory, output_file):
    # Liste pour stocker les données concaténées
    all_data = []

    # Parcourir tous les fichiers du répertoire d'entrée
    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            # Construire le chemin complet du fichier
            file_path = os.path.join(input_directory, filename)

            # Lire le contenu du fichier JSON
            with open(file_path, 'r') as f:
                annotations = json.load(f)

            if len(annotations) > 0:
                # Extraire le nom de l'image associé
                image_name = filename.replace(".json", ".jpg")

                # Créer un dictionnaire pour cette image
                data_entry = {
                    "image": image_name,
                    "annotations": annotations
                }

                # Ajouter ce dictionnaire à la liste des données
                all_data.append(data_entry)

            # Supprimer le fichier JSON après l'avoir lu
            os.remove(file_path)


    # Écrire toutes les données dans le fichier de sortie
    with open(output_file, 'w') as outfile:
        json.dump(all_data, outfile, indent=4)

    print(f"Fichier JSON concaténé généré : {output_file}")

def annotate(images_dir, filter):
    for pouet in 'train', 'test', 'valid':
        # Parcourir toutes les images du répertoire
        images_sub_dir = os.path.join(images_dir, pouet)
        for image_path in Path(images_sub_dir).rglob('*.jpg'):
            # Charger l'image
            img = Image.open(image_path)
            img_width, img_height = img.size
            img_name = os.path.splitext(image_path)[0]
            label = image_path.stem.split('_')[0]

            # Faire une prédiction
            results = model.predict(source=str(image_path))

            # Structure de données pour Create ML
            annotations = []

            # Parcourir chaque détection
            for result in results:
                for index, box in enumerate(result.boxes):  # x_min, y_min, x_max, y_max
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    # print(f"Classe: {class_name}, Confidence: {box.conf[0]:.2f}")
                    if class_name == filter and box.conf[0] > 0.6:
                        annotation = yolo_to_createml_format((img_width, img_height), box.xywh[0], label)
                        annotations.append(annotation)

            if len(annotations) == 0:
                os.remove(image_path)
                #print(f"Aucune annotations détectées pour {image_path}. Fichier supprimé.")
            else:
                # Générer le nom de fichier d'annotation
                annotation_filename = os.path.splitext(os.path.basename(image_path))[0] + '.json'
                annotation_filepath = os.path.join(images_sub_dir, annotation_filename)

                # Sauvegarder les annotations en JSON
                with open(annotation_filepath, 'w') as f:
                    json.dump(annotations, f, indent=4)

                # print(f"Annotations sauvegardées pour {image_path} sous {annotation_filepath}")

        compile_json_files(images_sub_dir, os.path.join(images_sub_dir, '_annotations.createml.json'))


def generate_dataset(video_dir, frames_dir, datasets_dir):
    # Example usage

    extract_frames_through_directories(video_dir, frames_dir, 10)

    # Appeler la fonction pour redimensionner toutes les images du répertoire
    resize_images_in_directory(frames_dir, (640, 1138))

    # Appeler la fonction pour augmenter toutes les images du répertoire
    augment_images_in_directory(frames_dir, frames_dir)

    # Appeler la fonction pour répartir les fichiers en un dataset

    ratios = [0.7, 0.2, 0.1]  # 70% train, 20% validation, 10% test
    split_frames(frames_dir, datasets_dir, ratios)



video_dir = '../videos/bottle'
frames_dir = '../tmp'
datasets_dir = '../out/datasets'

generate_dataset(video_dir, frames_dir, datasets_dir)
annotate(datasets_dir, 'bottle')
