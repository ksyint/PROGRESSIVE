import os
import cv2
from typing import List, Dict
from .core.config import Config
from .core.logger import get_logger
from .models.huatuo_wrapper import HuatuoWrapper
from .generators import EasyGenerator, MediumGenerator, HardGenerator
from .processors import ImageProcessor, MaskProcessor, BBoxProcessor
from .utils.modality import infer_modality
from .utils.validation import validate_detection_data

class DataGenerationPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger('DataGeneration')
        
        model_path = config.get('model.huatuo_path')
        self.model_wrapper = HuatuoWrapper(model_path)
        
        self.easy_gen = EasyGenerator(self.model_wrapper)
        self.medium_gen = MediumGenerator(self.model_wrapper)
        self.hard_gen = HardGenerator(self.model_wrapper)
        
        self.image_proc = ImageProcessor()
        self.mask_proc = MaskProcessor(config.get('processing.mask_threshold', 127))
        self.bbox_proc = BBoxProcessor()
    
    def process_single_item(self, data_item: Dict) -> List[Dict]:
        results = []
        
        image_path = data_item['image_path']
        anatomy_path = data_item.get('anatomy_path')
        annotations = data_item['annotations']
        
        if not os.path.exists(image_path):
            self.logger.warning(f"Image not found: {image_path}")
            return results
        
        img = cv2.imread(image_path)
        if img is None:
            self.logger.warning(f"Could not read image: {image_path}")
            return results
        
        h, w = img.shape[:2]
        
        organ_masks = {}
        if anatomy_path and os.path.exists(anatomy_path):
            organ_masks = self.mask_proc.load_organ_masks(anatomy_path, (h, w))
        
        modality = infer_modality(image_path)
        
        for ann in annotations:
            try:
                bbox_pixel = ann['bbox']
                bbox_data = self.bbox_proc.process_bbox(bbox_pixel, w, h)
                
                lesion_class = ann['class']
                
                host_organ = "unknown"
                if organ_masks:
                    host_organ = self.mask_proc.find_host_organ(
                        bbox_data['bbox_norm'], organ_masks
                    )
                
                seed = f"There is a {lesion_class} in the {host_organ}."
                
                easy_data = self.easy_gen.generate(
                    image_path, seed, lesion_class, host_organ, bbox_data['bbox_str']
                )
                
                medium_data = self.medium_gen.generate(
                    image_path, seed, lesion_class, host_organ, bbox_data['bbox_str']
                )
                
                result = {
                    'image_path': image_path,
                    'bbox': bbox_data['bbox_str'],
                    'question': easy_data.get('question', ''),
                    'cot': easy_data.get('cot', ''),
                    'answer': easy_data.get('answer', ''),
                    
                    'image_path2': image_path,
                    'question2': medium_data.get('question2', ''),
                    'cot2': medium_data.get('cot2', ''),
                    'answer2': medium_data.get('answer2', ''),
                    
                    'lesion_class': lesion_class,
                    'organ': host_organ,
                    'modality': modality
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing annotation: {e}")
                continue
        
        return results
    
    def process_detection_data(self, detection_data: List[Dict]) -> List[Dict]:
        validate_detection_data(detection_data)
        
        all_results = []
        
        for i, data_item in enumerate(detection_data):
            self.logger.info(f"Processing item {i+1}/{len(detection_data)}")
            
            results = self.process_single_item(data_item)
            all_results.extend(results)
        
        self.logger.info(f"Generated {len(all_results)} VQA samples")
        
        return all_results
