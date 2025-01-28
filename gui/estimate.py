# eu preciso disto porque o torch as vezes tenta usar a mps e eu nao tenho
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# isto continua a dar o warning mas pronto 
import warnings
warnings.filterwarnings("ignore", message="xFormers not available")

import cv2
import torch
import time
from ultralytics import YOLO

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depth_anything_v2.dpt import DepthAnythingV2




# para voces que têm gpus ou processador M1 ou M2, eu so tenho mesmo cpu :')
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# neste momento estamos a usar a versao small (vits) por causa de velocidade
# se quiserem experimentar modelos mais pesados go for it mas precisam do ficheiro .pth e coloca-lo em /checkpoints
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits'  
depth_model = DepthAnythingV2(**model_configs[encoder]) # carregar modelo estimacao distancias

# o ficheiro .pth tem os pre-trained weights de cada modelo;
# NOTA: podem mudar a location dos wieghts para a vossa GPU, será mais rápido. Tenham cuidado que com modelos mais pesados, pode sobrecarregar a gpu se nao tiverem assim tanta memoria. 
# basta mudarem 'cpu' para DEVICE
depth_model.load_state_dict(torch.load(f'../checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu')) 
depth_model = depth_model.to(DEVICE).eval() # mete o modelo na gpu/cpu e mete em modo eval para fazer as inferencias

yolo_model = YOLO('modelo_pessoas/best.pt')


def predict_depth(image):
    return depth_model.infer_image(image) 

def overlay_depth_values(image, depth, person_boxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    color = (255, 255, 255)  # branco 
    thickness = 1

    for box in person_boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        depth_value = depth[center_y, center_x] # distancia a que o centro da pessoa detetada esta da camera
        text = f"{depth_value:.1f}m"
        cv2.putText(image, text, (x1, y1 - 10), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return image



def process_frame(frame):
    start_time = time.time() # queria saber quanto demoravam cada parte, ver os prints
    
    depth = predict_depth(frame[:, :, ::-1])  # Converte BGR (openCV) para RGB (usado pelo modelo)

    '''
    as duas linhas a seguir servem para passar o valor de depth do modelo para um valor em metros

    o modelo é orientado para fazer um heat map, pintando com cores mais quentes o que está mais próximo e cores frias mais longe
    o modelo dá valores maiores à medida que as coisas se aproxima por isso é que faço 1/valor. O +e^-6 é para nao estar a dividir por zero
    ja devem ter percebido que os valores tambem nao variam em metros. Multipliquei por 10 para ajustar este rate. O 1.50 é porque preciasava deste offset
    estes valores (10  e 1.50) vieram de testes que fiz com o meu computador
    esta calibracao é um quick fix estou a tentar fazer algo melhor
    '''
    depth = 1/(depth +1e-6) 
    calibrated_depth = (depth * 10) - 1.50


    depth_time = time.time() 
    
    # modelo de detacao de pessoas
    
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

    overlay_with_values = overlay_depth_values(frame, calibrated_depth, person_boxes)
    overlay_time = time.time()
    
    print(f"Depth prediction time: {depth_time - start_time:.2f} seconds")
    print(f"YOLO detection time: {yolo_time - depth_time:.2f} seconds")
    print(f"Overlay time: {overlay_time - yolo_time:.2f} seconds")
    print(f"Total processing time: {overlay_time - start_time:.2f} seconds")
    
    return overlay_with_values

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read() # ret: boleano que indica se o frame foi capturado com sucesso
    if not ret:
        break

    result = process_frame(frame)
    cv2.imshow('Result', result)

    # 'q' para sair
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()