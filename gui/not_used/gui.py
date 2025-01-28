'''
Ignorar este ficheiro
isto foi uma tentativa lazy de estabilizar os valores para eles nao oscilarem tanto
basicamente estava a tentar que o valor só atualizasse depois de estar x segundos num determinado intervalo
nao mudou grande coisa
isto nem sequer é uma boa ideia
nao sei porque é que ainda tenho isto aqui 
'''


import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import warnings
warnings.filterwarnings("ignore", message="xFormers not available")

import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from collections import deque

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
FRAME_WINDOW = 10  # Number of frames to consider
RANGE_OFFSET = 0.05  # The range offset around the target value (e.g., ±0.05 for 6.2 gives [6.15, 6.24])

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits'  # Change this to the desired encoder
depth_model = DepthAnythingV2(**model_configs[encoder])
depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()



# Dictionary to store depth values for each detected person
depth_history = {}

def check_consistent_depth(depth_values, target_depth, offset):
    """Check if all depth values are within the range of the target depth ± offset."""
    min_depth = target_depth - offset
    max_depth = target_depth + offset
    return all(min_depth <= value <= max_depth for value in depth_values)

def overlay_depth_values_with_consistency_check(image, depth, boxes):
    global depth_history
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    color = (255, 255, 255)  # White color
    thickness = 1

    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        depth_value = depth[center_y, center_x]
        
        # Use the bounding box as a unique key to track depth history
        box_key = tuple(box)
        if box_key not in depth_history:
            depth_history[box_key] = deque(maxlen=FRAME_WINDOW)

        # Add the current depth value to the history
        depth_history[box_key].append(depth_value)

        # Determine the target depth (rounded to 1 decimal place)
        target_depth = round(depth_value, 1)

        # Check if the depth values are consistent around the target depth
        if check_consistent_depth(depth_history[box_key], target_depth, RANGE_OFFSET):
            display_depth = target_depth
        else:
            display_depth = depth_value

        # Display the depth value on the image
        text = f"{display_depth:.1f}m"
        cv2.putText(image, text, (x1, y1 - 10), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return image

def predict_depth(image):
    return depth_model.infer_image(image)



'''
def overlay_depth_on_image(image, depth):
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, depth_colored, 0.4, 0)
    return overlay
    
def overlay_depth_values(image, depth, boxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    color = (255, 255, 255)  # White color
    thickness = 1

    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        depth_value = depth[center_y, center_x]
        text = f"{depth_value:.1f}m"
        cv2.putText(image, text, (x1, y1 - 10), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return image
'''


def process_frame(frame):
    start_time = time.time()
    
    depth = predict_depth(frame[:, :, ::-1])  # Convert BGR to RGB
    depth = 1/(depth +1e-6)
    calibrated_depth = (depth * 10) - 1.50
    depth_time = time.time()
    
    # Load YOLO model and detect persons
    yolo_model = YOLO('/Users/gui/faculdade/ATLAS/dataset/recente/best.pt')
    results = yolo_model(frame)
    yolo_time = time.time()
    
    person_boxes = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            if r.names[int(cls)] == "person" and conf > 0.6:
                x1, y1, x2, y2 = map(int, box[:4])
                person_boxes.append([x1, y1, x2, y2])

    overlay_with_values = overlay_depth_values_with_consistency_check(frame, calibrated_depth, person_boxes)
    overlay_time = time.time()
    
    print(f"Depth prediction time: {depth_time - start_time:.2f} seconds")
    print(f"YOLO detection time: {yolo_time - depth_time:.2f} seconds")
    print(f"Overlay time: {overlay_time - yolo_time:.2f} seconds")
    print(f"Total processing time: {overlay_time - start_time:.2f} seconds")
    
    return overlay_with_values

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = process_frame(frame)
    cv2.imshow('Result', result)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()