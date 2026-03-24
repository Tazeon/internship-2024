from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2 as cv # import as cv

# Load YOLO model
model = YOLO("yolov8n.pt") # correct model path name


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
            box=coordinator, label=class_name, color=(255,0,0) # change box to blue color
        )
        # fix unbound error , make annotaor get in the loop

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model.predict(frame,classes = [15]) # list[15] is class cat

    for result in results:
        frame = draw_boxes(frame, result.boxes) # tab space to prevent syntax error

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    # remove already kub

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read() #.read() is correct function

        if ret:
            # Detect motorcycle from image frame
            frame_result = detect_object(frame)


            # Show result
            cv.namedWindow("Video", cv.WINDOW_NORMAL)
            cv.putText(frame,"Korawit Kaewkong-Clicknext-Internship-2026", (530,35) , cv.FONT_HERSHEY_SIMPLEX , 1, (0,0,255) , 2) #add name
            cv.imshow("Video", frame_result)
            cv.waitKey(30)

        else:
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv.destroyAllWindows()
