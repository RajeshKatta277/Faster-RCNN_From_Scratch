def calculate_iou(box, boxes):
    """
    Calculate Intersection over Union (IoU) between a box and multiple boxes.
    
    Inputs:
    - box: Numpy array of shape (4,) representing a single box [x1, y1, x2, y2].
    - boxes: Numpy array of shape (num_boxes, 4) containing multiple boxes.
    
    outputs:
    - iou: Numpy array of shape (num_boxes,) containing IoU values between the box and each box in boxes.
    """
    intersection_x1 = np.maximum(box[0], boxes[:, 0])
    intersection_y1 = np.maximum(box[1], boxes[:, 1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection_area = np.maximum(0, intersection_x2 - intersection_x1 + 1) * np.maximum(0, intersection_y2 - intersection_y1 + 1)
    
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    
    iou = intersection_area / (box_area + boxes_area - intersection_area)
    
    return iou
