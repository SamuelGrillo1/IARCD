import torch
from ultralytics import YOLO
import cv2
import os

#
weights_path = r'E:\Mills\ProjectMills\IARCD\fotoscolab\train\content\runs\segment\train5\weights\best.pt'
model = YOLO(weights_path)

folder_path = r'E:\Mills\ProjectMills\IARCD\datasetAdesivoruim\RCD-Adesivos-Ruins.v4i.yolov8\train\images'
save_folder = r'E:\Mills\ProjectMills\IARCD\Adesivos_Segmentados-dano'

# Criar pasta de salvamento se não existir
os.makedirs(save_folder, exist_ok=True)

# Percorrendo todas as imagens na pasta
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        results = model(image)

        # Obtendo as caixas delimitadoras
        boxes = results[0].boxes.xyxy

        # Verificando se há detecções
        if boxes.shape[0] > 0:
            # Obtendo a máscara de segmentação
            masks = results[0].masks.data

            if masks is not None:
                # Usando a máscara para extrair o adesivo segmentado
                mask = masks[0].cpu().numpy() > 0.5
                adesivo_segmentado = cv2.bitwise_and(image, image, mask=mask.astype('uint8') * 255)

                # Salvando o adesivo segmentado
                save_path = os.path.join(save_folder, filename)
                cv2.imwrite(save_path, adesivo_segmentado)

                print(f"Adesivo segmentado salvo em {save_path}")
            else:
                print(f"Nenhuma máscara encontrada em {filename}.")
        else:
            print(f"Nenhuma detecção encontrada em {filename}.")
