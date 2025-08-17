import cv2
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# This Enhances GPU Optimization
torch.backends.cudnn.benchmark = True

# Load lightweight YOLO model for real-time speed
# YOLOv8x is slow but more Accurate and Cuda here means Model will use GPU (faster than CPU) 
model = YOLO("yolov8x.pt").to('cuda')  
# This imports the data-set-Video
cap = cv2.VideoCapture("dataset_video (1).mp4") 

# Force output video properties to 1920x1080
# Extracting properties from input video , so that output video should also contain same properties.
width = 1920
height = 1080
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_combined.mp4", fourcc, fps, (width, height), True)


# Parameters for DBSCAN
DIST_THRESHOLD = 75
MIN_CROWD = 3

frame_number = 0
results_csv = []

# Create fullscreen or fixed size detection window
cv2.namedWindow("Crowd Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Crowd Detection", width, height)

# This will Make Crowd Detection to Full-Screen Mode
cv2.setWindowProperty("Crowd Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize input frame to match output resolution
    frame = cv2.resize(frame, (width, height))

    frame_number += 100

    #  Running YOLO detection on GPU for faster results
    results = model(frame, verbose=False)[0]  # verbose=False to remove console spam

    centroids = []
    boxes = []

    # Extract all person detections
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # Person class
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroids.append([cx, cy])
            boxes.append((int(x1), int(y1), int(x2), int(y2)))

    # Draw all detected people with green boxes + IDs
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {idx}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    crowd_count = 0
    people_in_crowds = 0

    if len(centroids) >= MIN_CROWD:
        X = np.array(centroids)
        clustering = DBSCAN(eps=DIST_THRESHOLD, min_samples=1).fit(X)
        labels = clustering.labels_

        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Ignore noise

            indices = np.where(labels == label)[0]
            if len(indices) >= MIN_CROWD:
                crowd_count += 1
                people_in_crowds += len(indices)

                # Overwrite green with red for crowd members
                for i in indices:
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"ID: {i+1}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Save group info
                results_csv.append({
                    "Frame Number": frame_number,
                    "Person Count in Crowd": len(indices)
                })

    # Display summary text on frame
    cv2.putText(frame, f"Crowds Detected: {crowd_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, f"People in Crowds: {people_in_crowds}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    # Write and display
    out.write(frame)
    cv2.imshow("Crowd Detection", frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Saving CSV file
df = pd.DataFrame(results_csv)
df.to_csv("crowd_detection_log.csv", index=False)

print("• Saved successfully:")
print("• output_combined.mp4")
print("• crowd_detection_log.csv")
