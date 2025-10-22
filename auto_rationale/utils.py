import numpy as np
import cv2
import os
from typing import List, Tuple, Dict



def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def load_organ_masks(anatomy_path, image_shape):
    organ_masks = {}
    
    if not os.path.exists(anatomy_path):
        return organ_masks
    
    mask_files = [f for f in os.listdir(anatomy_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for mask_file in mask_files:
        organ_name = os.path.splitext(mask_file)[0]
        mask_path = os.path.join(anatomy_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is not None:
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]))
            organ_masks[organ_name] = mask
    
    return organ_masks

def get_bounding_box_from_mask(mask, threshold):

    binary_mask = (mask > threshold).astype(np.uint8)
    
    if np.sum(binary_mask) == 0:
        return [0, 0, 0, 0]
    
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    h, w = mask.shape
    return [x_min/w, y_min/h, x_max/w, y_max/h]

def find_host_organ(lesion_box_norm: List[float], organ_masks: Dict[str, np.ndarray], threshold: int = 127) -> str:
    """ Finds the organ mask with the highest IoU with the NORMALIZED lesion box. """
    max_iou = 0
    best_organ = "unknown" # Default if no overlap or no masks

    # Ensure lesion_box_norm has 4 elements
    if len(lesion_box_norm) != 4:
        print(f"Warning: Invalid lesion_box_norm format: {lesion_box_norm}. Returning 'unknown'.")
        return best_organ

    for organ_name, mask in organ_masks.items():
        # Get organ bbox in normalized coordinates from the mask
        organ_box_norm = get_bounding_box_from_mask(mask, threshold)
        if sum(organ_box_norm) == 0: # Skip if mask was empty
            continue

        iou = calculate_iou(lesion_box_norm, organ_box_norm)

        if iou > max_iou:
            max_iou = iou
            best_organ = organ_name

    # Optional: Add a minimum IoU requirement?
    # if max_iou < config['processing']['iou_threshold']:
    #     return "unknown"

    return best_organ