import cv2
import numpy as np
import os

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

coco_names_path = "coco.names"

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
cap = cv2.VideoCapture('birds_yash2.mp4')

lower_color = np.array([0, 120, 70])  # Lower bound of the color in HSV
upper_color = np.array([10, 255, 255])  # Upper bound of the color in HSV

# Initialize variables for Camshift tracking
roi_hist = None
track_window = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

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

            if roi_hist is None:
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, lower_color, upper_color)
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                track_window = (x, y, w, h)

    if roi_hist is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.CamShift(dst, track_window, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break
print(os.getcwd())
cap.release()
cv2.destroyAllWindows()
