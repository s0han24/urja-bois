import os
import cv2
from ultralytics import YOLO

model = YOLO("models\\object detection\\best.pt")
names = model.names

crop_dir_name = "classify_by_brand_dataset"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)

idx = 0
for img_name in os.listdir("images2"):
    im0 = cv2.imread(os.path.join("images2", img_name))
    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            idx += 1

            crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

            cv2.imwrite(os.path.join(crop_dir_name, img_name + str(idx) + ".png"), crop_obj)