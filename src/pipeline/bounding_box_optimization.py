from math import tanh, exp
import numpy as np


def scaling(x, ceiling=3):
    return (1 - tanh(x * 2)) * ceiling


def get_object_bboxes(objects, img_size, padding_scale_ceiling=1):
    img_width = img_size['w'] - 1
    img_height = img_size['h'] - 1

    bboxes = []
    for object in objects:
        padding_w = scaling(object['w'] / img_width, padding_scale_ceiling) * object['w']
        padding_h = scaling(object['h'] / img_height, padding_scale_ceiling) * object['h']

        bboxes.append((
            max(object['y'] - padding_h, 0),
            max(object['x'] - padding_w, 0),
            min(object['y']+object['h']+padding_h, img_height),
            min(object['x']+object['w']+padding_h, img_width)
        ))
    return bboxes

def should_merge(box1, box2, overlap_threshold):
    YA1, XA1, YA2, XA2 = box1 
    YB1, XB1, YB2, XB2 = box2
    box1_area = (YA2 - YA1) * (XA2 - XA1)
    box2_area = (YB2 - YB1) * (XB2 - XB1)
    intersection_area = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))
    union_area = box1_area + box2_area - intersection_area

    if intersection_area / union_area > overlap_threshold:
        return True, (
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]),
        )

    return False, None

def merge_boxes(boxes, overlap_threshold):
    for k in range(len(boxes)):
        indices1, box1 = boxes[k]
        for l in range(k+1, len(boxes)):
            indices2, box2 = boxes[l]
            is_merge, new_box = should_merge(box1, box2, overlap_threshold)
            if is_merge:
                boxes[k] = None
                boxes[l] = (indices1.union(indices2), new_box)
                break

    boxes = [b for b in boxes if b]
    return boxes 

def get_pair_bboxes(objects, merge_threshold = 0.7):
    num_objects = len(objects)
    bbox_indices = np.full([num_objects, num_objects], -1)
    
    joined_bboxes = []
    for i in range(num_objects):
        for j in range(i+1, num_objects):
            object1 = objects[i]
            object2 = objects[j]

            joined_bboxes.append(({(i,j)}, (
                min(object1['y'], object2['y']),
                min(object1['x'], object2['x']),
                max(object1['y'] + object1['h'], object2['y'] + object2['h']),
                max(object1['x'] + object1['w'], object2['x'] + object2['w']),
            )))

    merged_boxes = merge_boxes(joined_bboxes, merge_threshold)
    for k, (indices, _) in enumerate(merged_boxes):
        for i,j in indices:
            bbox_indices[i,j] = k
            bbox_indices[j,i] = k

    return [box for _, box in merged_boxes], bbox_indices