from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
yoloModel = YOLO('yolov8n.pt')

# Display model information (optional)
yoloModel.info()
print(yoloModel.names)
img = PIL.Image.open('y4.png')

result = yoloModel(img)
print(result[0].names)
for box in result[0].boxes:
  if result[0].names[box.cls.item()] == "traffic light":
    xyxy = box.xyxy.tolist()
    print(xyxy)
    img = img.crop(xyxy[0])
    print(img)
    display(img)
#print(result[0].boxes.xyxyn)
#

from roboflow import Roboflow
rf = Roboflow(api_key="")
project = rf.workspace("seb-bulba").project("roadsigns_2")
version = project.version(2)
dataset = version.download("yolov8")
