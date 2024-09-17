import coremltools as ct
import onnx

# Charger le modèle ONNX simplifié
onnx_model = onnx.load("best_simplified.onnx")

# Convertir le modèle ONNX en modèle Core ML
coreml_model = ct.converters.onnx.convert(onnx_model, minimum_ios_deployment_target='13')

# Sauvegarder le modèle Core ML (.mlmodel)
coreml_model.save("YOLOv8.mlmodel")
