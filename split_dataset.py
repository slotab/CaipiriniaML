import os
import shutil
import random

HOME = os.getcwd()
print(HOME)


def repartition_data(base_dir, train_ratio=0.7, valid_ratio=0.25, test_ratio=0.05):
    # Définir les chemins des répertoires
    images_dir = os.path.join(base_dir, 'all', 'images')
    labels_dir = os.path.join(base_dir, 'all', 'labels')
    
    train_images_dir = os.path.join(base_dir, 'train', 'images')
    train_labels_dir = os.path.join(base_dir, 'train', 'labels')
    valid_images_dir = os.path.join(base_dir, 'valid', 'images')
    valid_labels_dir = os.path.join(base_dir, 'valid', 'labels')
    test_images_dir = os.path.join(base_dir, 'test', 'images')
    test_labels_dir = os.path.join(base_dir, 'test', 'labels')
    
    # Créer les répertoires de sortie s'ils n'existent pas
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    
    # Lister toutes les images dans le répertoire "images"
    all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Mélanger les fichiers pour une répartition aléatoire
    random.shuffle(all_images)
    
    # Calculer le nombre de fichiers pour chaque répertoire
    total_images = len(all_images)
    train_count = int(total_images * train_ratio)
    valid_count = int(total_images * valid_ratio)
    test_count = total_images - train_count - valid_count  # Le reste va dans le test

    # Diviser les images en trois groupes : train, valid et test
    train_images = all_images[:train_count]
    valid_images = all_images[train_count:train_count + valid_count]
    test_images = all_images[train_count + valid_count:]
    
    # Copier les images et annotations associées dans les répertoires train/valid/test
    for image_file in train_images:
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        
        # Copier dans le dossier train
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(train_images_dir, image_file))
        shutil.copy(os.path.join(labels_dir, label_file), os.path.join(train_labels_dir, label_file))
    
    for image_file in valid_images:
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        
        # Copier dans le dossier valid
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(valid_images_dir, image_file))
        shutil.copy(os.path.join(labels_dir, label_file), os.path.join(valid_labels_dir, label_file))
    
    for image_file in test_images:
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        
        # Copier dans le dossier test
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(test_images_dir, image_file))
        shutil.copy(os.path.join(labels_dir, label_file), os.path.join(test_labels_dir, label_file))
    
    print(f"Répartition terminée : {len(train_images)} images dans 'train', {len(valid_images)} images dans 'valid', et {len(test_images)} images dans 'test'.")


# Exemple d'utilisation
base_dir = f"{HOME}/dataset"  
repartition_data(base_dir, train_ratio=0.7, valid_ratio=0.25, test_ratio=0.05)

