import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import re

class CurriculumDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.domain_map = {}
        
        for idx, item in enumerate(self.data):
            lesion_class = item["lesion_class"]
            modality = item['modality'] 
            domain = f"{lesion_class}_{modality}"
            
            if domain not in self.domain_map:
                self.domain_map[domain] = []
            self.domain_map[domain].append(idx)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_path = item['image_path']  
        image_path2 = item['image_path2']  
        image_path3 = item.get('image_path3', image_path2)
        
        bbox = item['bbox']
        
        cot = item['cot']
        cot2 = item['cot2']
        
        answer = item['answer']  
        answer2 = item['answer2']  
        answer3 = item.get('answer3', answer2)
        
        question = item["question"]
        question2 = item["question2"]
        question3 = item.get("question3", question2)
        
        lesion_class = item['lesion_class']
        modality = item["modality"]
        domain = f"{lesion_class}_{modality}"
        
        organ = item["organ"]
        rationale = f"{lesion_class}_{organ}"
        
        return {
            'image_path': image_path,
            'image_path2': image_path2,
            'image_path3': image_path3,
            
            'bbox': bbox,
            
            'cot': cot,
            'cot2': cot2,
            
            "answer": answer,
            "answer2": answer2,
            "answer3": answer3,
            
            "question": question,
            "question2": question2,
            "question3": question3,
            
            'domains': domain,
            'rationale': rationale
        }
    
    def get_domains(self):
        return list(self.domain_map.keys())
    
    def get_domain_indices(self, domain):
        return self.domain_map.get(domain, [])
    
    def set_stage(self, stage):
        self.stage = stage
