from ultralytics import YOLO
import cv2

# Load your trained gun detection model
gun_model = YOLO('C:/Users/sai jaswanth/runs/detect/train8/weights/last.pt')

# Load the pre-trained YOLOv8 model for person detection (COCO dataset)
person_model = YOLO('yolov8n.pt')  # Nano version for speed

# Initialize the video capture for webcam (use DirectShow backend for better compatibility)
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the webcam opened successfully
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Class ID for 'person' in the COCO dataset is 0
person_class_id = 0

while True:
    # Capture the frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Run gun detection model on the current frame
    gun_results = gun_model.predict(frame, conf=0.5)  # Adjust confidence threshold if needed

    # Draw bounding boxes for detected guns
    for result in gun_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]

            # Assuming class_id == 0 is 'handgun'
            if class_id == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for gun
                cv2.putText(frame, f'Handgun: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Run person detection model on the current frame
    person_results = person_model.predict(frame, conf=0.5)  # Adjust confidence threshold if needed

    # Draw bounding boxes for detected people
    for result in person_results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get class id
            if class_id == person_class_id:  # Only process if detected object is 'person'
                confidence = box.conf[0]  # Get confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                # Draw bounding box around the detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for person
                cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame with the bounding boxes for both gun and person detections
    cv2.imshow('Gun and Person Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
