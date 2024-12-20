import cv2
from ultralytics import YOLO

# Load YOLOv8 model pre-trained on the COCO dataset
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for the nano version, which is faster

# Initialize the video capture (0 for webcam, or use a video file path)
video_capture = cv2.VideoCapture(0)

# Class ID for 'person' in COCO dataset is 0
person_class_id = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Frame not captured properly.")
        break

    # Ensure the frame is in the correct format (e.g., BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLOv8 model on the current frame
    try:
        results = model(frame_rgb, conf=0.5)  # Adjust confidence threshold if needed

        # Loop through the detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])  # Get class id
                if class_id == person_class_id:  # Only process if detected object is 'person'
                    confidence = box.conf[0]    # Get confidence score
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                    # Draw bounding box around the detected person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    except Exception as e:
        print(f"Error during model inference: {e}")
        break

    # Display the resulting frame
    cv2.imshow('YOLOv8 Person Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
