from ultralytics import YOLO
import os
import cv2

###
# DEMO SCRIPT
###

HOME = os.getcwd()
print(HOME)

# CHOOSE A VIDEO 
INPUT_VIDEO_PATH = f"{HOME}/videos/IMG_7193.MOV"
# TRAIN A MODEL TO MODEL GENER
TRAINED_MODEL_PATH = f"{HOME}/runs/detect/caipirinia/weights/best.pt"

model = YOLO(TRAINED_MODEL_PATH)


# Open the video file
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
