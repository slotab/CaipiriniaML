from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/Users/bslota/IdeaProjects/summer-challenge-ia/runs/yolov8_caipirinia_model3/weights/best.pt")

# Export the model to CoreML format
model.export(format="coreml")  # creates 'yolov8n.mlpackage'
