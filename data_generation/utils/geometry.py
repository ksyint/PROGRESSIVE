import numpy as np
from typing import List

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

def normalize_bbox(bbox_pixel: List[int], image_width: int, image_height: int) -> List[float]:
    x1, y1, x2, y2 = bbox_pixel
    return [
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height
    ]

def denormalize_bbox(bbox_norm: List[float], image_width: int, image_height: int) -> List[int]:
    x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm
    return [
        int(x1_norm * image_width),
        int(y1_norm * image_height),
        int(x2_norm * image_width),
        int(y2_norm * image_height)
    ]

def bbox_to_string(bbox: List[float], precision: int = 4) -> str:
    format_str = f"{{:.{precision}f}}"
    return "[" + ", ".join([format_str.format(x) for x in bbox]) + "]"

def validate_bbox(bbox: List[float]) -> bool:
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    
    if not all(isinstance(x, (int, float)) for x in bbox):
        return False
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    return True

def bbox_area(bbox: List[float]) -> float:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def bbox_center(bbox: List[float]) -> tuple:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)
