from ultralytics import YOLO
import os
import glob

# Load a model
model = YOLO('runs/detect/train16/weights/best.pt') # load a custom model

# Collect all image paths in your directory
image_paths = glob.glob('/project/ultralytics-main/ultralytics-main/ultralytics/RCD.v5i.yolov8/test/images/*.jpg')

# Predict on the images
results = model(image_paths)

# Print the results
for result in results:
    if len(result.boxes) > 0:
        print(f'Detections found in image {result.path}')

# Create a directory for the annotated images
save_dir = "/project/ultralytics-main/ultralytics-main/ultralytics/fotosresultsteste"
os.makedirs(save_dir, exist_ok=True)

# Perform the test and save the annotated images
for result in results:
    if len(result.boxes) > 0:
        for box in result.boxes.xyxy:
            with open(os.path.join(save_dir, f"{os.path.basename(result.path)}.txt"), 'a') as f:
                f.write(f"{box}\n")
