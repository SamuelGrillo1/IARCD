import cv2
import os
import numpy as np
from ultralytics import YOLO
import torch

def is_inside(box1, box2):
    """Verifica se o centro de box1 está dentro de box2."""
    cx1, cy1 = (box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2
    return box2[0] <= cx1 <= box2[2] and box2[1] <= cy1 <= box2[3]

def calculate_area(box):
    x1, y1, x2, y2 = box[:4]
    return (x2 - x1) * (y2 - y1)

def calculate_damage_severity(adesivo_box, danos_inside_boxes):
    adesivo_area = calculate_area(adesivo_box)
    total_damage_area = sum([calculate_area(dano) for dano in danos_inside_boxes])

    damage_ratio = total_damage_area / adesivo_area

    if damage_ratio < 0.05:
        return "Leve"
    elif damage_ratio < 0.1:
        return "Medio"
    else:
        return "Severo"

def process_frame(frame):
    results = model(frame)
    result = results[0]
    detected_boxes = result.boxes.data.cpu().numpy()

    detected_masks = None if result.masks is None else result.masks.data.cpu().numpy()

    adesivo_counter = 0
    adesivo_longo_counter = 0
    total_danos = 0

    danos_severidade_info = []

    for i, box in enumerate(detected_boxes):
        x1, y1, x2, y2, score, class_id = map(int, box)
        label = class_labels[class_id]

        if class_id == 0 or class_id == 1:
            danos_inside_boxes = [dano_box for dano_box in detected_boxes if int(dano_box[5]) == 2 and is_inside(dano_box[:4], box[:4])]
            severity = calculate_damage_severity(box, danos_inside_boxes)
            danos_inside = len(danos_inside_boxes)
            
            if class_id == 0:
                label += f"-{adesivo_longo_counter} tem {danos_inside} danos com severidade {severity}"
                adesivo_longo_counter += 1
            else:
                label += f"-{adesivo_counter} tem {danos_inside} danos com severidade {severity}"
                adesivo_counter += 1

            total_danos += danos_inside
            danos_severidade_info.append((danos_inside, severity))

        if detected_masks is not None:
            mask = detected_masks[i]
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            overlay = frame.copy()
            overlay[mask_resized > 0.5] = segmentation_color[:3]
            alpha = 0.5
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
        cv2.putText(frame, label, (x1, y1 - 10), text_font, text_size, text_color, text_thickness)

    return frame, total_danos, danos_severidade_info

torch.set_default_tensor_type('torch.FloatTensor')
weights_path = r'E:\Mills\ProjectMills\IARCD\fotoscolab\train\content\runs\segment\train5\weights\best.pt'
model = YOLO(weights_path)

folder_path = r'E:\Mills\ProjectMills\IARCD\datacolab\train\train\images'
save_folder = r'E:\Mills\ProjectMills\IARCD\testecodeadesivo\fotos'
os.makedirs(save_folder, exist_ok=True)

class_labels = {
    0: "Adesivo Longo",
    1: "Adesivos",
    2: "dano"
}

box_color =  (2,64,195)       
text_color = (255,255,255)
segmentation_color = (33,112,243)


box_thickness = 2
text_font = cv2.FONT_HERSHEY_PLAIN
text_size = 1.2
#text_size = 3
text_thickness = 2


for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    save_path = os.path.join(save_folder, filename)

    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = cv2.imread(file_path)
        processed_image, total_danos, danos_severidade_info = process_frame(image)
        cv2.imwrite(save_path, processed_image)
        print(f"Imagem processada salva em {save_path}")
        print(f"Total de danos na imagem: {total_danos}")
        for i, (qtd, severity) in enumerate(danos_severidade_info):
            print(f"Adesivo {i+1}: Quantidade de danos: {qtd}, Severidade: {severity}")

    elif filename.endswith('.MOV') or filename.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                processed_frame, _, _ = process_frame(frame)
                out.write(processed_frame)
            else:
                break

        cap.release()
        out.release()
        print(f"Vídeo processado salvo em {save_path}")
