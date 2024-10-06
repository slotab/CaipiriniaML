import os

HOME = os.getcwd()
print(HOME)

def update_class_id(filepath, original_class_id, target_class_id):
    with open(filepath, 'r') as f:
        lignes = f.readlines()

    # Ouverture du fichier en mode écriture après lecture
    with open(filepath, 'w') as f:
        for ligne in lignes:
            # Divise la ligne en parties
            elements = ligne.split()

            # Remplace le premier chiffre '0' par '22' si c'est le cas
            if elements[0] == original_class_id:
                elements[0] = target_class_id
            
            # Écrire la ligne modifiée dans le fichier
            f.write(' '.join(elements) + '\n')

def update_class_id_in_folder(source_dir, original_class_id, target_class_id):
    # Parcourt tous les fichiers texte dans le répertoire source
    for fichier in os.listdir(source_dir):
        if fichier.endswith(".txt"):
            fichier_chemin = os.path.join(source_dir, fichier)
            update_class_id(fichier_chemin, original_class_id, target_class_id)
            print(f"Traitement effectué sur : {fichier}")


update_class_id_in_folder(f"{HOME}/dataset/train/labels", '0', '22')
