from ultralytics import YOLO
import os

HOME = os.getcwd()
print(HOME)


# Load a pretrained YOLO11n model
model = YOLO(f"{HOME}/yolo11n.pt")

# Run inference on the source
results = model(f"{HOME}/images/images_IMG_7010.jpg")  # list of Results objects