from sh_stuff import connect
import cv2
from ultralytics import YOLO

model_logo = YOLO("models\\brand logo\\best.pt")
model_object = YOLO("models\\shelf detection\\best.pt")

names = model_logo.names

def draw_bounding_boxes(image, results):
    # Extract bounding boxes, classes, names, and confidences

    i = 0
    # Iterate through the results
    for box1, box2 in results:
        x_min_obj, y_min_obj, x_max_obj, y_max_obj = box1
        x_min_logo, y_min_logo, x_max_logo, y_max_logo = box2
        
        image = cv2.rectangle(image, (int(x_min_obj), int(y_min_obj)), (int(x_max_obj), int(y_max_obj)), (i, i, i), 5)
        image = cv2.rectangle(image, (int(x_min_logo), int(y_min_logo)), (int(x_max_logo), int(y_max_logo)), (i, i, i), 5)
        # Optionally, put class_id text
        image = cv2.putText(image, 'logo',(int(x_min_logo), int(y_min_logo - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        i+=10

    # image = cv2.flip(image, 1)
    # Display the image
    cv2.imshow('frame', image)

    

while True:
    frame = cv2.imread("can.jpeg")
    results_logos = model_logo(frame)
    results_objects = model_object(frame)
    results = connect(results_objects[0].boxes.xyxy.tolist(), results_logos[0].boxes.xyxy.tolist(), 0.5)
    draw_bounding_boxes(frame, results)
    if cv2.waitKey(1) == ord('q'):
        break


