import cv2
import os
import shutil
import random
import json
from pathlib import Path
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import yaml

# Charger le modèle YOLOv8
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


def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def augment_image(image_path, output_directory, base_filename):
    """Générer des variations de luminosité et de contraste pour une image"""
    with Image.open(image_path) as img:
        for i in range(3):  # Par exemple, 3 variations de luminosité
            brightness_factor = random.uniform(0.5, 1.5)
            bright_image = adjust_brightness(img, brightness_factor)
            bright_image.save(
                os.path.join(output_directory, f"{base_filename}_brightness_{i}.jpg")
            )
            print(
                f"Image avec variation de luminosité générée : {base_filename}_brightness_{i}.jpg"
            )

        for i in range(3):  # Par exemple, 3 variations de contraste
            contrast_factor = random.uniform(0.5, 1.5)
            contrast_image = adjust_contrast(img, contrast_factor)
            contrast_image.save(
                os.path.join(output_directory, f"{base_filename}_contrast_{i}.jpg")
            )
            print(
                f"Image avec variation de contraste générée : {base_filename}_contrast_{i}.jpg"
            )


def augment_images_in_directory(root_directory, output_directory):
    """Appliquer des augmentations à toutes les images dans le répertoire donné"""
    os.makedirs(output_directory, exist_ok=True)
    for subdir, _, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                base_filename, ext = os.path.splitext(file)
                augment_image(file_path, output_directory, base_filename)


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
    source_directory, destination_directory, frame_interval=10
):
    """Extraire des frames de toutes les vidéos dans un répertoire source"""
    os.makedirs(destination_directory, exist_ok=True)
    index = 1

    for dirpath, dirnames, filenames in os.walk(source_directory):
        for filename in filenames:
            source_file_path = os.path.join(dirpath, filename)
            if source_file_path.endswith(".MOV"):
                extract_frames(
                    source_file_path,
                    destination_directory,
                    dirpath.split("/")[-1],
                    index,
                    frame_interval,
                )
                print(f"Analysé : {source_file_path}")
                index += 1


def convert_to_yolo_format(img_width, img_height, box):
    x_center, y_center, width, height = box
    x_center_normalized = x_center / img_width
    y_center_normalized = y_center / img_height
    width_normalized = width / img_width
    height_normalized = height / img_height
    return x_center_normalized, y_center_normalized, width_normalized, height_normalized


def annotate(images_dir, filter, classes):
    """Annoter les images en utilisant le modèle YOLOv8 et sauvegarder les annotations au format YOLO"""
    for image_path in Path(images_dir).rglob("*.jpg"):
        img = Image.open(image_path)
        img_width, img_height = img.size
        img_name = os.path.splitext(image_path)[0]
        label = image_path.stem.split("_")[0]

        results = model.predict(source=str(image_path))
        yolo_annotations = []

        for result in results:
            for index, box in enumerate(result.boxes):
                yolo_class_id = int(box.cls[0])
                target_class_id = classes.index(label)  # int(box.cls[0])
                class_name = model.names[yolo_class_id]
                if class_name == filter and box.conf[0] > 0.6:
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
            annotation_filename = (
                os.path.splitext(os.path.basename(image_path))[0] + ".txt"
            )
            annotation_filepath = os.path.join(images_dir, annotation_filename)
            with open(annotation_filepath, "w") as f:
                f.write("\n".join(yolo_annotations))
            print(
                f"Annotations sauvegardées pour {image_path} sous {annotation_filepath}"
            )


def generate_dataset(video_dir, frames_dir):
    """Générer un dataset au format YOLOv8"""
    extract_frames_through_directories(video_dir, frames_dir, 10)
    resize_images_in_directory(frames_dir, (360, 640))
    augment_images_in_directory(frames_dir, frames_dir)


def create_yolo_directories(output_directory):
    """Créer les sous-répertoires train, valid, et test."""
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(output_directory, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_directory, split, "labels"), exist_ok=True)


def move_files(image_list, src_img_dir, src_lbl_dir, dest_img_dir, dest_lbl_dir):
    """Déplacer les fichiers d'image et leurs annotations dans les répertoires YOLO."""
    for image_file in image_list:
        # Chemins des fichiers
        img_src_path = os.path.join(src_img_dir, image_file)
        lbl_src_path = os.path.join(
            src_lbl_dir, image_file.replace(".jpg", ".txt")
        )  # Remplace jpg par txt pour trouver le fichier d'annotation

        # Chemins de destination
        img_dest_path = os.path.join(dest_img_dir, image_file)
        lbl_dest_path = os.path.join(dest_lbl_dir, image_file.replace(".jpg", ".txt"))

        # Déplacer les fichiers image et annotation
        shutil.move(img_src_path, img_dest_path)
        if os.path.exists(lbl_src_path):  # Vérifie si le fichier d'annotation existe
            shutil.move(lbl_src_path, lbl_dest_path)
        else:
            print(f"Annotation manquante pour {image_file}")


def split_dataset(image_dir, annotation_dir, output_directory, ratios=(0.7, 0.2, 0.1)):
    """Diviser les fichiers image et annotation en train, valid et test."""
    create_yolo_directories(output_directory)

    # Obtenir la liste des fichiers d'image
    all_images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    random.shuffle(all_images)  # Mélange les fichiers pour une répartition aléatoire

    # Calculer les indices pour les coupes
    total_images = len(all_images)
    train_idx = int(total_images * ratios[0])
    valid_idx = train_idx + int(total_images * ratios[1])

    # Diviser les fichiers en train, valid, test
    train_files = all_images[:train_idx]
    valid_files = all_images[train_idx:valid_idx]
    test_files = all_images[valid_idx:]

    # Déplacer les fichiers dans leurs répertoires respectifs
    move_files(
        train_files,
        image_dir,
        annotation_dir,
        os.path.join(output_directory, "train", "images"),
        os.path.join(output_directory, "train", "labels"),
    )
    move_files(
        valid_files,
        image_dir,
        annotation_dir,
        os.path.join(output_directory, "valid", "images"),
        os.path.join(output_directory, "valid", "labels"),
    )
    move_files(
        test_files,
        image_dir,
        annotation_dir,
        os.path.join(output_directory, "test", "images"),
        os.path.join(output_directory, "test", "labels"),
    )

    print(
        f"Répartition terminée : {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test."
    )


def generate_yaml(classes, yaml_output_path):
    """
    Génère un fichier YAML pour le dataset YOLOv8.

    :param output_directory: Chemin vers le répertoire contenant les répertoires 'train', 'valid', 'test'.
    :param dataset_name: Nom du dataset.
    :param classes: Liste des classes du dataset.
    :param yaml_output_path: Chemin de sortie pour le fichier YAML.
    """

    # Chemins des sous-répertoires d'images
    train_images_dir = os.path.join("./train", "images")
    val_images_dir = os.path.join("./valid", "images")
    test_images_dir = os.path.join("./test", "images")

    # Générer la structure du fichier YAML
    yaml_content = {
        "train": train_images_dir,
        "val": val_images_dir,
        "test": test_images_dir,
        "nc": len(classes),  # Nombre de classes
        "names": classes,  # Noms des classes
    }

    # Sauvegarder le fichier YAML
    with open(yaml_output_path, "w") as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)

    print(f"Fichier YAML généré : {yaml_output_path}")


def get_class_names(root_directory):
    """
    Cette fonction parcourt un répertoire donné et renvoie une liste des noms de sous-répertoires.

    :param root_directory: Le chemin du répertoire à parcourir.
    :return: Une liste contenant les noms des sous-répertoires.
    """
    # Vérifier si le répertoire existe
    if not os.path.isdir(root_directory):
        print(f"Erreur : Le répertoire {root_directory} n'existe pas.")
        return []

    # Obtenir la liste des sous-répertoires
    classes_names = [
        name.split("_")[0]
        for name in os.listdir(root_directory)
        if os.path.isdir(os.path.join(root_directory, name))
    ]
    # classes_names = []
    # for name in os.listdir(root_directory):
    #     full_path = os.path.join(root_directory, name)
    #     if os.path.isdir(full_path):
    #         # Diviser par '_' et garder la première partie (le 'prefix')
    #         prefix = name.split('_')[0]
    #         classes_names.append(prefix)

    return list(dict.fromkeys(classes_names))


def lire_fichier_en_liste(nom_fichier):
    # Ouvrir le fichier en mode lecture
    with open(nom_fichier, "r") as fichier:
        # Lire toutes les lignes du fichier et les retourner sous forme de liste
        lignes = fichier.readlines()
    # Supprimer les retours à la ligne (\n) à la fin de chaque ligne
    lignes = [ligne.strip() for ligne in lignes]
    return lignes


# def generate_yaml():
#     train: path/to/dataset/images/train/  # Chemin vers les images d'entraînement
# val: path/to/dataset/images/val/      # Chemin vers les images de validation
#
# # Dictionnaire des classes
# names:
# 0: aperol  # Exemple de nom de classe
# 1: gin
# 2: vodka

video_dir = "../videos/short"
frames_dir = "../tmp"
datasets_dir = "datasets"

nom_du_fichier = "caiprinia_labels.txt"
classes = lire_fichier_en_liste(nom_du_fichier)
# classes = get_class_names(video_dir)
# classes = ["cocacola", "aperol", "gin", "rum", "curacao", "vodka", "cointreau", "tonic", "martini", "campari"]

generate_dataset(video_dir, frames_dir)

annotate(frames_dir, "bottle", classes)

# split_dataset(frames_dir, frames_dir, datasets_dir, ratios=(0.7, 0.2, 0.1))

# yaml_output_path = os.path.join(datasets_dir, 'caipirinia.yaml')

generate_yaml(classes, "caipirinia.yaml")
