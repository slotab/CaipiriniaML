from PIL import Image
import pytesseract
import os

HOME = os.getcwd()
print(HOME)

# Chemin vers l'exécutable Tesseract (sous Windows, par exemple)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def lire_texte_image(chemin_image):
    # Charger l'image
    image = Image.open(chemin_image)
    
    # Utiliser Tesseract pour extraire le texte
    texte = pytesseract.image_to_string(image)
    
    return texte

# Exemple d'utilisation
texte_extrait = lire_texte_image(f"{HOME}/images/images_IMG_7010.jpg")
print(texte_extrait)