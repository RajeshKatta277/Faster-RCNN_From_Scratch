def non_max_suppression(boxes, scores, threshold):
    """
    Apply Non-Maximum Suppression to filter out overlapping boxes.
    
    Args:
    - boxes: Numpy array of shape (num_boxes, 4) containing bounding boxes in the format [x1, y1, x2, y2].
    - scores: Numpy array of shape (num_boxes, 1) containing confidence scores for each box.
    - threshold: Threshold value for IoU (Intersection over Union).
    
    Returns:
    - selected_indices: List of indices of the selected boxes after NMS.
    """
    sorted_indices = np.argsort(scores.flatten())[::-1]
    selected_indices = []
    
    while len(sorted_indices) > 0:
        best_box_index = sorted_indices[0]
        selected_indices.append(best_box_index)
        best_box = boxes[best_box_index]
        other_boxes = boxes[sorted_indices[1:]]
        iou = calculate_iou(best_box, other_boxes)
        filtered_indices = np.where(iou <= threshold)[0]
        sorted_indices = sorted_indices[filtered_indices + 1]  # adding 1 because of slicing
        
    return selected_indices

