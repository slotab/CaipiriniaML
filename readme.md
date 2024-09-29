# CaipirinIA Model Training

Scripts for prepare datasets and train model for our iOs App


## Via CreateML

todo

## Via YoloV8


### Python env

```
python -m venv caipirinienv
source caipirinienv/bin/activate
```

Install dependencies
```
pip install ultralytics
```

### build datasets from videos

The folder should have the following structure:
/rum
    videos with only one bottle of this kind
/gin_brandA
    videos with only one bottle of this kind
/gin_brandB
    videos with only one bottle of this kind
etc...

!! The first part of the parent folder name (before '_') will be the class name in the model


```
python videos_to_yolo_datasets.py
```

### train the model

```
python yolo_train.py
```
or with pre-trained model
```
yolo detect train data=caipirinia.yaml model=yolov8n.pt epochs=10 imgsz=640 device=0
```
or from scratch
```
yolo detect train data=caipirinia.yaml model=yolov8n.yaml epochs=10 imgsz=640 device=0
```

### convert to coreml 

```
python export_yolo.py
```
or
```
yolo export model=yolov8_caipirinia.pt format=coreml
```
!!? need python 3.11

--- 

## Autodistill

```
pip install autodistill autodistill-grounded-sam autodistill-yolov8
```

