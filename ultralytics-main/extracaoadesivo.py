import torch
from ultralytics import YOLO
import cv2
import os

weights_path = r'E:\Mills\ProjectMills\IARCD\fotoscolab\train\content\runs\segment\train5\weights\best.pt'
model = YOLO(weights_path)

folder_path = r'E:\Mills\ProjectMills\IARCD\datasetAdesivoruim\RCD-Adesivos-Ruins.v4i.yolov8\valid\images'
save_folder = r'E:\Mills\ProjectMills\IARCD\Adesivos_Segmentados-dano'

# Criar pasta de salvamento se não existir
os.makedirs(save_folder, exist_ok=True)

# Percorrendo todas as imagens na pasta
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        results = model(image)
        print(results)

        # Obtendo as caixas delimitadoras e os nomes das classes
        detection_results = results[0]
        boxes = detection_results.boxes.xyxy.cpu().numpy()
        names = detection_results.names

        # Verificando se há detecções
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                class_idx = int(box[-1])
                if class_idx in names:
                    class_name = names[class_idx]
                    if class_name in ["Adesivo Longo", "Adesivos"]:
                        danos_no_adesivo = 0

                        # Verificando se algum dano está dentro desta caixa delimitadora
                        for j, other_box in enumerate(boxes):
                            other_class_idx = int(other_box[-1])
                            if other_class_idx in names:
                                other_class_name = names[other_class_idx]
                                if other_class_name == "dano" and all(box[:2] <= other_box[:2]) and all(box[2:] >= other_box[2:]):
                                    danos_no_adesivo += 1

                        print(f"Adesivo {i} ({class_name}) tem {danos_no_adesivo} danos.")
                else:
                    print(f"Índice de classe desconhecido {class_idx} em {filename}.")
        else:
            print(f"Nenhuma detecção encontrada em {filename}.")
