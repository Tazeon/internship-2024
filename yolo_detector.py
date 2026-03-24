"""
I created two versions of the code: yolo_detector.py and yolo_detector_no_trackingline.py,
because I am unsure about the instruction in the README.md that says "Draw a tracking line for the detected cat."

If this instruction means that I need to implement a tracking line, it would require a lot of changes to the
draw_boxes and detect_object functions. However, I am not certain whether modifying these parts would still fully
comply with the intended requirements.

If making those changes is acceptable, I will submit yolo_detector.py. 
Otherwise, I will submit yolo_detector_no_trackingline.py.

P.S. I am not sure whether the instruction "Add your name + Clicknext-Internship-2024 to the top-right corner" requires keeping "2024" exactly as specified.
However, since I am submitting this in 2026, I updated it to "Clicknext-Internship-2026."

Thank you.
"""

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2 as cv # import as cv
from collections import defaultdict
import numpy as np

# Load YOLO model
model = YOLO("yolov8n.pt") # correct model path name

# Store the track history
track_history = defaultdict(lambda: [])


def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)

    if boxes.id is None:
        return annotator.result()
    # Handle in case there no object is detected

    track_ids = boxes.id.int().tolist()

    for box, track_id in zip(boxes, track_ids):
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf

        # Draw bounding box
        annotator.box_label(
            box=coordinator, label=class_name, color=(255,0,0) # change box to blue color
        )
        # fix unbound error , make annotaor get in the loop

        """ 
        Implement and Modiflying from Multi-Object Tracking with Ultralytics YOLO
        https://docs.ultralytics.com/modes/track/#plotting-tracks-over-time
        """
        x, y, w, h = box.xywh[0].tolist()
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 30 tracks for 30 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=5) 

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame

    ### use .track() instead of predict because this instruction need to have tracking line
    results = model.track(frame,persist=True,classes = [15]) # list[15] is class cat

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
