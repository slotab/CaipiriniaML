##
# AUTO LABELING
##

from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

# from autodistill_yolov8 import YOLOv8

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
base_model = GroundedSAM(
    ontology=CaptionOntology(
        {
            "blue bottle": "gin",
            "red can": "cocacola",
            "transparent bottle with yellow label": "tonic",
            "lime": "lime",
            "banana": "banana",
        }
    )
)

# label all images in a folder called `context_images`
base_model.label(input_folder="./images", output_folder="./datasetg")

# target_model = YOLOv8("yolov8n.pt")
# target_model.train("./dataset/data.yaml", epochs=200)

# run inference on the new model
# pred = target_model.predict("./dataset/valid/your-image.jpg", confidence=0.5)
# print(pred)
