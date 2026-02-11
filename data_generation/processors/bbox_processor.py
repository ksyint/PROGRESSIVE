from typing import List

class InvalidBBoxError(Exception):
    pass

class BBoxProcessor:
    def __init__(self):
        pass
    
    def process_bbox(self, bbox_pixel: List[int], image_width: int, image_height: int) -> dict:
        from ..utils.geometry import normalize_bbox, bbox_to_string, validate_bbox
        
        try:
            bbox_norm = normalize_bbox(bbox_pixel, image_width, image_height)
            
            if not validate_bbox(bbox_norm):
                raise InvalidBBoxError(f"Invalid bbox after normalization: {bbox_norm}")
            
            bbox_str = bbox_to_string(bbox_norm)
            
            return {
                'bbox_pixel': bbox_pixel,
                'bbox_norm': bbox_norm,
                'bbox_str': bbox_str
            }
        except Exception as e:
            raise InvalidBBoxError(f"Error processing bbox {bbox_pixel}: {e}")
    
    def batch_process_bboxes(self, bboxes: List[List[int]], image_width: int, image_height: int) -> List[dict]:
        results = []
        for bbox in bboxes:
            try:
                result = self.process_bbox(bbox, image_width, image_height)
                results.append(result)
            except InvalidBBoxError as e:
                print(f"Skipping invalid bbox: {e}")
                continue
        return results
