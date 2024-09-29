from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/Users/bslota/IdeaProjects/summer-challenge-ia/caipirinia/runs/detect/yolov8_caipirinia6/weights/best.pt")

# Export the model to CoreML format
model.export(format="coreml")
