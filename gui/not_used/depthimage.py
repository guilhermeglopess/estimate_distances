'''
Ignorar este ficheiro
Primeiros testes, so com uma imagem
'''




import warnings
warnings.filterwarnings("ignore", message="xFormers not available")

import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

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

def predict_depth(image):
    return depth_model.infer_image(image)

'''
def overlay_depth_on_image(image, depth):
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, depth_colored, 0.4, 0)
    return overlay
'''


def overlay_depth_values(image, depth, boxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # White color
    thickness = 1

    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        depth_value = depth[center_y, center_x]
        text = f"{depth_value:.2f}m"
        cv2.putText(image, text, (x1, y1 - 10), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return image

def process_image(image_path):
    start_time = time.time()
    
    image_load_start_time = time.time()
    image = cv2.imread(image_path)
    image_load_end_time = time.time()
    
    read_time = time.time()
    
    depth = predict_depth(image[:, :, ::-1])  # Convert BGR to RGB
    depth_time = time.time()
    
    # Load YOLO model and detect persons
    yolo_model = YOLO('/Users/gui/faculdade/ATLAS/dataset/recente/best.pt')
    results = yolo_model(image)
    yolo_time = time.time()
    
    person_boxes = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if r.names[int(cls)] == "person":
                x1, y1, x2, y2 = map(int, box[:4])
                person_boxes.append([x1, y1, x2, y2])

    overlay_with_values = overlay_depth_values(image, depth, person_boxes)
    overlay_time = time.time()
    
    print(f"Image load time: {image_load_end_time - image_load_start_time:.2f} seconds")
    print(f"Image read time: {read_time - start_time:.2f} seconds")
    print(f"Depth prediction time: {depth_time - read_time:.2f} seconds")
    print(f"YOLO detection time: {yolo_time - depth_time:.2f} seconds")
    print(f"Overlay time: {overlay_time - yolo_time:.2f} seconds")
    print(f"Total processing time: {overlay_time - start_time:.2f} seconds")
    
    return overlay_with_values

# Example usage
image_path = '/Users/gui/faculdade/ATLAS/dataset/image_9_27_jpg.rf.5b4642ba7b02e4013d5fb624265e97bf.jpg'
input("Press Enter to estimate depth...")
result = process_image(image_path)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()