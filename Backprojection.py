import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
  


# Load class names
coco_names_path = "/Users/yogendrasinghrawat/Desktop/DRDO/VARIOUS TYPES OF TRACKING/coco.names"

try:
    with open(coco_names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: The file {coco_names_path} was not found.")
    classes = []
except Exception as e:
    print(f"Error: An unexpected error occurred while reading {coco_names_path}: {e}")
    classes = []

# Initialize video capture
cap = cv2.VideoCapture('BP.mp4')

# Initialize variables for tracking and back projection
tracker_initialized = False
tracker = None
roi_hist = None  # Histogram for back projection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analyze the outputs
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:  # You can adjust the threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Initialize tracker and back projection on the first detected object
            if not tracker_initialized:
                tracker = cv2.TrackerMIL_create()  # For OpenCV < 4.5.1
                tracker.init(frame, tuple(boxes[i]))
                tracker_initialized = True

                # Set up the ROI for tracking and calculate histogram
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Update tracker and calculate speed
    if tracker_initialized:
        ret, box = tracker.update(frame)
        if ret:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            # Back projection using the histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            
            # Apply a mask and bitwise operation to highlight the back projection
            mask = cv2.merge([back_proj, back_proj, back_proj])
            result = cv2.bitwise_and(frame, mask)
            
            cv2.imshow('Back Projection', result)

    # Display the result
    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
