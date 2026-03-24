from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2 as cv

# Load YOLO model
model = YOLO("yolov8n.pt")


def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf

    # Draw bounding box
    annotator.box_label(
        box=coordinator, label=class_name, color=colors(class_id, True)
    )

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model.predict(frame,classes = [15])

    for result in results:
        frame = draw_boxes(frame, result.boxes)

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    # remove already kub

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()

        if ret:
            # Detect motorcycle from image frame
            frame_result = detect_object(frame)


            # Show result
            cv.namedWindow("Video", cv.WINDOW_NORMAL)
            cv.imshow("Video", frame_result)
            cv.waitKey(30)

        else:
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv.destroyAllWindows()
