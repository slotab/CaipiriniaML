import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Charger le modèle MobileNetV2 pré-entraîné
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Prétraitement de l'image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir BGR vers RGB
    image_resized = cv2.resize(image, (224, 224))   # Redimensionner pour correspondre à l'entrée du modèle MobileNetV2
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_resized)
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension supplémentaire pour le batch
    return image, image_array

# Faire des prédictions sur l'image
def predict_image(image_array):
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Afficher les prédictions
def display_predictions(image, predictions):
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predictions: {predictions[0][1]} ({predictions[0][2]*100:.2f}%)\n"
              f"{predictions[1][1]} ({predictions[1][2]*100:.2f}%)\n"
              f"{predictions[2][1]} ({predictions[2][2]*100:.2f}%)")
    plt.show()

# Chemin de l'image à analyser
image_path = './../images/images_IMG_7010.jpg'

# Exécuter le pipeline de détection
image, image_array = preprocess_image(image_path)
predictions = predict_image(image_array)
display_predictions(image, predictions)
