import cv2
import supervision as sv
from ultralytics import YOLO

# Open webcam (use 0 for the default webcam)
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load YOLO model
model = YOLO("yolov8s.pt")

# Initialize the box annotator
bbox_annotator = sv.BoxAnnotator()

while video.isOpened():
    ret, frame = video.read()
    if ret:
        # Run YOLO inference
        results = model(frame)
        result = results[0]

        # Convert YOLO detections to Supervision Detections
        detections = sv.Detections.from_ultralytics(result)

        # Filter detections with confidence > 0.5
        detections = detections[detections.confidence > 0.5]

        # Create labels for each detection (class name + confidence score)
        labels = [
            f"{result.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate the frame with bounding boxes and labels
        frame = bbox_annotator.annotate(scene=frame, detections=detections)

        # Add labels manually (if annotate no longer handles them)
        for box, label in zip(detections.xyxy, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Error: Unable to read frame.")
        break

# Release resources
video.release()
cv2.destroyAllWindows()
