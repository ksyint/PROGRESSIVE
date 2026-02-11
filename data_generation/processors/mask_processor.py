import cv2
import numpy as np
import os
from typing import Dict

class MaskProcessingError(Exception):
    pass

class MaskProcessor:
    def __init__(self, mask_threshold: int = 127):
        self.mask_threshold = mask_threshold
    
    def load_organ_masks(self, anatomy_path: str, image_shape: tuple) -> Dict[str, np.ndarray]:
        organ_masks = {}
        
        if not os.path.exists(anatomy_path):
            return organ_masks
        
        mask_files = [f for f in os.listdir(anatomy_path) 
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for mask_file in mask_files:
            organ_name = os.path.splitext(mask_file)[0]
            mask_path = os.path.join(anatomy_path, mask_file)
            
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if mask is not None:
                    mask = cv2.resize(mask, (image_shape[1], image_shape[0]))
                    organ_masks[organ_name] = mask
            except Exception as e:
                raise MaskProcessingError(f"Error loading mask {mask_path}: {e}")
        
        return organ_masks
    
    def get_bounding_box_from_mask(self, mask: np.ndarray) -> list:
        binary_mask = (mask > self.mask_threshold).astype(np.uint8)
        
        if np.sum(binary_mask) == 0:
            return [0, 0, 0, 0]
        
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        h, w = mask.shape
        return [x_min/w, y_min/h, x_max/w, y_max/h]
    
    def find_host_organ(self, lesion_bbox: list, organ_masks: Dict[str, np.ndarray]) -> str:
        from ..utils.geometry import calculate_iou
        
        max_iou = 0
        best_organ = "unknown"
        
        if len(lesion_bbox) != 4:
            return best_organ
        
        for organ_name, mask in organ_masks.items():
            organ_bbox = self.get_bounding_box_from_mask(mask)
            
            if sum(organ_bbox) == 0:
                continue
            
            iou = calculate_iou(lesion_bbox, organ_bbox)
            
            if iou > max_iou:
                max_iou = iou
                best_organ = organ_name
        
        return best_organ
