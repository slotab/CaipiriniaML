from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/Users/bslota/IdeaProjects/summer-challenge-ia/caipirinia/runs/detect/yolov8_caipirinia6/weights/best.pt")



# Export the model to CoreML format
model = model.export(format="coreml")

class_labels = [
    'campari',
    'aperol']

#model.class_labels = class_labels

model.input_description["input_1"] = "Input image to be classified"
model.output_description["classLabel"] = "Most likely image category"

# Set model author name
model.author = '"Beubeu du 54'

# Set the license of the model
model.license = "Please see https://github.com/tensorflow/tensorflow for license information, and https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet for the original source of the model."

# Set a short description for the Xcode UI
model.short_description = "Detects the dominant objects present in an image from a set of 1001 categories such as trees, animals, food, vehicles, person etc. The top-1 accuracy from the original publication is 74.7%."

# Set the preview type
model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "objectDetector"

# Set a version for the model
model.version = "2.0"


# Save the model
model.save("CaiprinIA.mlpackage")
