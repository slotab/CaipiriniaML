import os
from PIL import Image


HOME = os.getcwd()
print(HOME)

def reduce_image_size(input_dir, output_dir, target_size=(800, 600), quality=85):
    # Vérifier si le répertoire de sortie existe, sinon le créer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parcourir toutes les images du répertoire d'entrée
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Ouvrir l'image
            with Image.open(input_path) as img:
                # Redimensionner l'image tout en conservant le ratio
                img.thumbnail(target_size)
                
                # Sauvegarder l'image redimensionnée dans le répertoire de sortie
                img.save(output_path, quality=quality, optimize=True)
            
            print(f"{filename} a été redimensionnée et sauvegardée dans {output_dir}")


# Exemple d'utilisation
input_directory = f"{HOME}/screenshots/demo"
output_directory = f"{HOME}/screenshots/demo2"
reduce_image_size(input_directory, output_directory)