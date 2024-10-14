from ultralytics import YOLO
import os
import cv2
from pathlib import Path
from PIL import Image
import json
import shutil
import yaml
import glob

##############################
# MAIN SCRIPT OF THE PROJECT !!
##############################

HOME = os.getcwd()
print(HOME)


# Charger le modèle
model = YOLO("yolov8n.pt")


def resize_image(image_path, target_size=(640, 640)):
    """Redimensionner l'image pour correspondre à la taille cible"""
    with Image.open(image_path) as img:
        resized_img = img.resize(target_size)
        resized_img.save(image_path)
        print(f"Image redimensionnée : {image_path}")


def resize_images_in_directory(root_directory, target_size=(640, 640)):
    """Redimensionner toutes les images dans le répertoire donné"""
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                try:
                    resize_image(file_path, target_size)
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {file_path} : {e}")


def extract_frames(video_path, output_dir, prefix, index, frame_interval=10):
    """Extraire des frames d'une vidéo"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_index = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret:
            if frame_index % frame_interval == 0:
                frame_filename = os.path.join(
                    output_dir, f"{prefix}_{index:03d}_{saved_frame_count:04d}.jpg"
                )
                cv2.imwrite(frame_filename, frame)
                print(f"Saved {frame_filename}")
                saved_frame_count += 1
            frame_index += 1
        else:
            break

    cap.release()
    print(f"Extraction terminée. Frames sauvegardées dans {output_dir}")


def extract_frames_through_directories(
    source_directory, prefix, destination_directory, frame_interval=10
):
    """Extraire des frames de toutes les vidéos dans un répertoire source"""
    os.makedirs(destination_directory, exist_ok=True)
    index = 1

    for dirpath, dirnames, filenames in os.walk(source_directory):
        for filename in filenames:
            source_file_path = os.path.join(dirpath, filename)
            if source_file_path.endswith(".MOV"):
                # prefix = dirpath.split('/')[-1]
                extract_frames(
                    source_file_path,
                    destination_directory,
                    prefix,
                    index,
                    frame_interval,
                )
                print(f"Extracted : {source_file_path}")
                index += 1


def move_files(source_dir, destination_dir, pattern="*"):
    # Vérifie si le répertoire de destination existe, sinon le crée
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Parcourt tous les fichiers dans le répertoire source qui correspondent au pattern
    fichiers_a_deplacer = glob.glob(os.path.join(source_dir, pattern))

    for fichier in fichiers_a_deplacer:
        # Extraire le nom de fichier depuis le chemin complet
        filename = os.path.basename(fichier)
        destination_path = os.path.join(destination_dir, filename)

        # Déplace le fichier
        shutil.move(fichier, destination_path)
        print(f"Déplacé: {filename} vers {destination_dir}")


def videos_to_frames(video_path, prefix, frames_dir, index, target_size=(360, 640)):
    extract_frames(video_path, frames_dir, prefix, index, 10)
    print(f"Extracted : {video_path}")

    resize_images_in_directory(frames_dir, target_size)
    # augment_images_in_directory(frames_dir, frames_dir)


def convert_to_yolo_format(img_width, img_height, box):
    x_center, y_center, width, height = box
    x_center_normalized = x_center / img_width
    y_center_normalized = y_center / img_height
    width_normalized = width / img_width
    height_normalized = height / img_height
    return x_center_normalized, y_center_normalized, width_normalized, height_normalized


def process_videos(source_directory, class_labels, out_dir):
    """Extraire des frames de toutes les vidéos dans un répertoire source"""
    os.makedirs(out_dir, exist_ok=True)
    index = 1
    for dirpath, dirnames, filenames in os.walk(source_directory):
        for filename in filenames:
            source_file_path = os.path.join(dirpath, filename)
            if source_file_path.endswith(".json"):
                # on convert.json file detected
                convert_json_filepath = os.path.join(dirpath, "convert.json")
                if os.path.exists(convert_json_filepath):
                    convert_json = load_convert_file(convert_json_filepath)
                    class_label_map = convert_json["class_label"]
                    class_label_from = class_label_map["from"]
                    class_label_to = class_label_map["to"]
                    confidence = class_label_map.get("confidence", 0.7)                   
                    
                    print(f"Repertoire : {dirpath}")
                    
                    annotate_videos(dirpath, class_labels, class_label_from, class_label_to, confidence, out_dir)
                    index += 1


def annotate_videos(video_dir, class_labels, class_label_from, class_label_to, confidence, out_dir):
    index = 1
    for video_path in Path(video_dir).rglob("*.MOV"):
        # folder_name_as_label = os.path(images_dir).stem.split("_")[0]
        videos_to_frames(video_path, class_label_to, f"{HOME}/tmp", index)
        target_class_id = class_labels.index(class_label_to)

        annotate_images(f"{HOME}/tmp", class_label_from, target_class_id, confidence)
        move_files(f"{HOME}/tmp", f"{out_dir}/images", "*.jpg")
        move_files(f"{HOME}/tmp", f"{out_dir}/labels", "*.txt")
        index += 1


def annotate_images(images_dir, class_label_from, target_class_id, confidence):
    for image_path in Path(images_dir).rglob("*.jpg"):
        annotate_image(image_path, class_label_from, target_class_id, confidence)


def annotate_image(image_path, class_label_from, target_class_id, confidence=0.7):
    # Annoter les images en utilisant le modèle YOLOv8
    # et sauvegarder les annotations au format YOLO
    img = Image.open(image_path)
    img_width, img_height = img.size

    results = model.predict(source=str(image_path))
    yolo_annotations = []

    for result in results:
        for index, box in enumerate(result.boxes):
            model_class_id = int(box.cls[0])
            model_class_name = model.names[model_class_id]
            # target_class_id = class_labels.index(class_label_to)
            if model_class_name in class_label_from and box.conf[0] > confidence:
                x_center, y_center, width, height = convert_to_yolo_format(
                    img_width, img_height, box.xywh[0]
                )
                yolo_annotations.append(
                    f"{target_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

    if len(yolo_annotations) == 0:
        os.remove(image_path)
        print(f"Aucune annotation détectée pour {image_path}. Fichier supprimé.")
    else:
        annotation_filepath = os.path.splitext(image_path)[0] + ".txt"
        with open(annotation_filepath, "w") as f:
            f.write("\n".join(yolo_annotations))
        print(f"Annotations sauvegardées pour {image_path} sous {annotation_filepath}")


def load_class_labels(nom_fichier):
    # Ouvrir le fichier en mode lecture
    with open(nom_fichier, "r") as fichier:
        # Lire toutes les lignes du fichier et les retourner sous forme de liste
        lignes = fichier.readlines()
    # Supprimer les retours à la ligne (\n) à la fin de chaque ligne
    lignes = [ligne.strip() for ligne in lignes]
    return lignes


def load_convert_file(chemin_fichier):
    # Ouvrir le fichier JSON en mode lecture
    with open(chemin_fichier, "r", encoding="utf-8") as fichier:
        # Charger le contenu du fichier JSON dans un objet Python
        donnees = json.load(fichier)

    return donnees


def generate_yaml(classes, train_images_dir, val_images_dir, yaml_output_path):
    yaml_content = {
        "train": train_images_dir,
        "val": val_images_dir,
        # "test": test_images_dir,
        "nc": len(classes),  # Nombre de classes
        "names": classes,  # Noms des classes
    }

    # Sauvegarder le fichier YAML
    with open(yaml_output_path, "w") as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)

    print(f"Fichier YAML généré : {yaml_output_path}")


class_labels = load_class_labels("caipirinia_labels.txt")


process_videos(f"{HOME}/../videos/others", class_labels, f"{HOME}/dataset/new")
