from ultralytics import YOLO
import torch
# Validação
model = YOLO('/usr/src/ultralytics/ultralytics-main/runs/detect/train31/weights/best.pt')
model.val()
# Teste 
source = '/usr/src/ultralytics/ultralytics-main/ultralytics/RCD.v5i.yolov8/test/images/frame_18739_jpg.rf.68f1a92447ecf47a70eabcf14442901b.jpg'
results = model(source)