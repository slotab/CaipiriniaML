# 1.  Configurer l’accès à l’API Google Drive :
# •	Va sur la console Google Cloud.
# •	Crée un nouveau projet.
# •	Active l’API Google Drive pour ce projet.
# •	Crée des identifiants OAuth 2.0 et télécharge le fichier credentials.json.
# 2.	Installer les bibliothèques Python nécessaires :
# •	Installe pydrive en utilisant pip :
# ```
# pip install pydrive
# ```

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

# Authentification Google
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Cela ouvrira un navigateur pour autoriser l'accès à votre compte Google Drive

drive = GoogleDrive(gauth)

# ID du dossier à télécharger
folder_id = 'TON_DOSSIER_ID'  # Remplace par l'ID de ton dossier

# Créer un répertoire local pour enregistrer les fichiers
local_folder = 'downloaded_folder'
if not os.path.exists(local_folder):
    os.makedirs(local_folder)

# Récupérer la liste des fichiers dans le dossier Google Drive
file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

# Télécharger chaque fichier du dossier
for file in file_list:
    print(f"Téléchargement du fichier : {file['title']}")
    file.GetContentFile(os.path.join(local_folder, file['title']))

print("Téléchargement terminé.")
