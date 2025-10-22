import json
import cv2
import os
import re # Added
from typing import Dict, List
from utils import load_organ_masks, find_host_organ
from Huatuo.cli import HuatuoChatbot 



def _infer_modality(image_path):
    path_lower = image_path.lower()
    if 'xray' in path_lower or 'radiograph' in path_lower:
        return 'xray'
    elif 'ct' in path_lower:
        return 'ct'
    elif 'mri' in path_lower or 'mr' in path_lower:
        return 'mri'
    else:
        return 'unknown' 

def use_huatuo(huatuogpt_vision_model_path, query, image_path):
   
    image_path_list=[image_path] 
    bot = HuatuoChatbot(huatuogpt_vision_model_path)
    output = bot.inference(query, image_path_list)
    return output


class MedCLMDataGenerator:
    def __init__(self, huatuo_path: str):
        self.model_path = huatuo_path
       

    def _parse_huatuo_response(self, response: str, keys: List[str]) -> Dict:
        parts = response.split('|')
        result = {}
        for i, key in enumerate(keys):
            try:
              
                content = parts[i].split(':', 1)[1].strip() if ':' in parts[i] else parts[i].strip()
                result[key] = content
            except IndexError:
                print(f"Warning: Could not parse part {i+1} for key '{key}' from response: {response}")
                result[key] = "" 
        return result

    def generate_easy_data(self, image_path, seed, lesion_class, organ, box_str):
        prompt = f"""
        Given this medical image showing: {seed}.
        The {lesion_class} is suspected within the box region {box_str} which is [xmin, ymin, xmax, ymax].
        (xmin, ymin) is top-left, (xmax, ymax) is bottom-right.

        Generate a specific diagnostic question targeting the highlighted region.
        Provide: 1) The question, 2) Step-by-step reasoning including the box {box_str}, 3) A concise final answer.

        Format exactly as: Question: [question] | Reasoning: [reasoning including {box_str}] | Answer: [answer]
        """
        response = use_huatuo(self.model_path, prompt, image_path)
        return self._parse_huatuo_response(response, ["question", "cot", "answer"])

    def generate_medium_data(self, image_path, seed, lesion_class, organ, box_str):
        prompt = f"""
        Analyze the entire medical image, noting the finding: {seed}.
        The {lesion_class} is located in the {organ} around region {box_str} [xmin, ymin, xmax, ymax].

        Generate a question about identifying abnormalities in the whole image.
        Provide: 1) The question, 2) Reasoning describing the finding's location (mentioning {box_str}), 3) A final answer including location context.

        Format exactly as: Question: [question] | Reasoning: [reasoning including {box_str}] | Answer: [answer]
        """
        response = use_huatuo(self.model_path, prompt, image_path)
        parsed = self._parse_huatuo_response(response, ["question2", "cot2", "answer2"])
        return parsed



    def process_detection_data(self, detection_data: List[Dict], output_type: str = "vqa", mask_threshold: int = 127) -> List[Dict]:
        results = []
       
        if output_type != "vqa":
             print(f"Warning: output_type '{output_type}' not fully supported for 3-stage generation. Defaulting to VQA-like structure.")

        for i, data_item in enumerate(detection_data):
            print(f"Processing item {i+1}/{len(detection_data)}...")
            image_path = data_item["image_path"]
            anatomy_path = data_item.get("anatomy_path") 
            annotations = data_item["annotations"]
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found {image_path}, skipping.")
                continue
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}, skipping.")
                continue

            organ_masks = {}
            if anatomy_path and os.path.exists(anatomy_path):
                 organ_masks = load_organ_masks(anatomy_path, img.shape[:2])
            else:
                 print(f"Warning: Anatomy path '{anatomy_path}' not found or not provided for {image_path}. Organ finding might be inaccurate.")


            modality = _infer_modality(image_path) 

            for j, annotation in enumerate(annotations):
                print(f"  Processing annotation {j+1}/{len(annotations)}...")
                
                bbox_pixel = annotation["bbox"]
                h, w = img.shape[:2]
                try:
                    x1_norm = bbox_pixel[0] / w
                    y1_norm = bbox_pixel[1] / h
                    x2_norm = bbox_pixel[2] / w
                    y2_norm = bbox_pixel[3] / h
                    bbox_str_norm = f"[{x1_norm:.4f}, {y1_norm:.4f}, {x2_norm:.4f}, {y2_norm:.4f}]"
                    bbox_norm_list = [x1_norm, y1_norm, x2_norm, y2_norm]
                except Exception as e:
                    print(f"Error converting bbox {bbox_pixel} for {image_path}: {e}. Skipping annotation.")
                    continue

                lesion_class = annotation["class"]
                
                host_organ = "unknown" 
                if organ_masks:
                    host_organ = find_host_organ(bbox_norm_list, organ_masks, threshold=mask_threshold)

                seed = f"There is a {lesion_class} in the {host_organ}."
                
                
                try:
                    easy_data = self.generate_easy_data(image_path, seed, lesion_class, host_organ, bbox_str_norm)
                    medium_data = self.generate_medium_data(image_path, seed, lesion_class, host_organ, bbox_str_norm)

                    final_result = {
                        "image_path": image_path, 
                        "bbox": bbox_str_norm,
                        "question": easy_data.get("question", ""),
                        "cot": easy_data.get("cot", ""),
                        "answer": easy_data.get("answer", ""),

                        "image_path2": image_path, 
                        "question2": medium_data.get("question2", ""),
                        "cot2": medium_data.get("cot2", ""),
                        "answer2": medium_data.get("answer2", ""),

                        "lesion_class": lesion_class,
                        "organ": host_organ,
                        "modality": modality
                    }
                    results.append(final_result)
                except Exception as e:
                    print(f"Error during VLM generation for {image_path}, annotation {j+1}: {e}")
                    continue

        return results