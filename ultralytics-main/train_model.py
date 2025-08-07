#from ultralytics import YOLO

#model = YOLO("yolov8n.pt") 

#model.train(data="/project/ultralytics-main/ultralytics-main/ultralytics/RCD.v5i.yolov8/data.yaml", epochs=10 )

#metrics = model.val()

#results = model("/project/ultralytics-main/ultralytics-main/ultralytics/IMG_4557.png")

#print(results)

from ultralytics import YOLO
import torch
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

args = dict(data=r"E:\Mills\ProjectMills\IARCD\YoloV8\ultralytics-main\ultralytics-main\ultralytics\RCD.v5i.yolov8\data.yaml", epochs=30, imgsz=640, task='detect', batch=8)
model_variant = "yolov8n"

model = YOLO('yolov8n.pt')
# Load the YOLOv8 model
#model = YOLO(f"{model_variant}.pt")  # load a pretrained model

# Train the model using our arguments from before
# If running remotely they may have been changed by ClearML
results = model.train(**args)