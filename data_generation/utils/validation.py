import os
from typing import Dict, List

class ValidationError(Exception):
    pass

def validate_detection_data(data: List[Dict]) -> bool:
    if not isinstance(data, list):
        raise ValidationError("Detection data must be a list")
    
    for i, item in enumerate(data):
        if 'image_path' not in item:
            raise ValidationError(f"Item {i} missing 'image_path'")
        
        if 'annotations' not in item:
            raise ValidationError(f"Item {i} missing 'annotations'")
        
        if not isinstance(item['annotations'], list):
            raise ValidationError(f"Item {i} 'annotations' must be a list")
        
        for j, ann in enumerate(item['annotations']):
            if 'bbox' not in ann:
                raise ValidationError(f"Item {i}, annotation {j} missing 'bbox'")
            
            if 'class' not in ann:
                raise ValidationError(f"Item {i}, annotation {j} missing 'class'")
            
            bbox = ann['bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValidationError(f"Item {i}, annotation {j} has invalid bbox format")
    
    return True

def validate_output_data(data: List[Dict]) -> bool:
    required_fields = [
        'image_path', 'bbox', 'question', 'cot', 'answer',
        'image_path2', 'question2', 'cot2', 'answer2',
        'lesion_class', 'organ', 'modality'
    ]
    
    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                raise ValidationError(f"Output item {i} missing required field '{field}'")
    
    return True

def validate_image_path(image_path: str) -> bool:
    if not os.path.exists(image_path):
        raise ValidationError(f"Image not found: {image_path}")
    
    if not os.path.isfile(image_path):
        raise ValidationError(f"Path is not a file: {image_path}")
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    _, ext = os.path.splitext(image_path)
    
    if ext.lower() not in valid_extensions:
        raise ValidationError(f"Invalid image extension: {ext}")
    
    return True
