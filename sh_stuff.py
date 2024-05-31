def sh_score(box1, box2):
    x1_1, y1_1, x1_2, y1_2 = box1
    x2_1, y2_1, x2_2, y2_2 = box2

    left_x = max(x1_1, x2_1)
    right_x = min(x1_2, x2_2)
    top_y = max(y1_1, y2_1)
    bottom_y = min(y1_2, y2_2)

    area1 = (x1_2 - x1_1) * (y1_2 - y1_1)
    intersect_area = (right_x - left_x) * (bottom_y - top_y)

    return intersect_area / area1

def connect(object_boxes, brand_boxes, threshold):
    best_boxes = []
    for object_box in object_boxes:
        best_sh_score = 0
        best_brand_box = None
        for brand_box in brand_boxes:
            if sh_score(brand_box, object_box) > best_sh_score:
                best_brand_box = brand_box 
                best_sh_score = sh_score(brand_box, object_box)
        if best_sh_score >= threshold:
            best_boxes.append((object_box, best_brand_box))
    return best_boxes
