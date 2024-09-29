import cv2
import supervision as sv

from inference.models.yolo_world.yolo_world import YOLOWorld

image_path = "images/images_IMG_7027.jpg"

image = cv2.imread(image_path)

model = YOLOWorld(model_id="yolo_world/l")
classes = ["bottle", "lime", "mint", "sugar", "ice", "soda", "rum"]
results = model.infer(image_path, text=classes, confidence=0.03)

detections = sv.Detections.from_inference(results)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [classes[class_id] for class_id in detections.class_id]

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections
)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels
)

sv.plot_image(annotated_image)
