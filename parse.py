# The following code is a parser that takes a single annotation file in image class x1 y1 x2 y2 format 
# and converts it to YOLO format

# file structure:
#   -<annotation file>
#   -images(folder)
#   -parse.py
#   -split.py
import cv2
import pandas as pd

annotation_file = 'flickr_logos_27_dataset_training_set_annotation.txt'

df = pd.read_csv(annotation_file, header=None, sep=' ')
classes = list((df[1]).unique())

d = {}
for i in range(len(classes)):
    d[classes[i]] = i

print(classes)

with open(annotation_file) as file:
    for line in file:
        img_name = str(line.split(' ')[0])
        class_name = str(line.split(' ')[1])
        xmin = int(line.split(' ')[3])
        ymin = int(line.split(' ')[4])
        xmax = int(line.split(' ')[5])
        ymax = int(line.split(' ')[6])

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (xmin + xmax) / 2 
        b_center_y = (ymin + ymax) / 2
        b_width    = (xmax - xmin)
        b_height   = (ymax - ymin)
        
        # Normalise the co-ordinates by the dimensions of the image
        image = cv2.imread('images/' + img_name)
        image_h, image_w, image_c = image.shape
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        class_id = d[class_name]
        path = 'images\\'+img_name.removesuffix('.jpg')+'.txt'
        with open(path,'a') as label:
            text = ' '.join([str(class_id),str(b_center_x),str(b_center_y),str(b_width),str(b_height)])
            label.write(text+'\n')