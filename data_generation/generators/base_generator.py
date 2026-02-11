from abc import ABC, abstractmethod
from typing import Dict

class BaseGenerator(ABC):
    def __init__(self, model_wrapper):
        self.model = model_wrapper
    
    @abstractmethod
    def generate(self, image_path: str, seed: str, lesion_class: str, 
                organ: str, bbox_str: str) -> Dict[str, str]:
        pass
    
    def parse_response(self, response: str, keys: list) -> Dict[str, str]:
        parts = response.split('|')
        result = {}
        
        for i, key in enumerate(keys):
            try:
                content = parts[i].split(':', 1)[1].strip() if ':' in parts[i] else parts[i].strip()
                result[key] = content
            except IndexError:
                result[key] = ""
        
        return result
    
    def create_prompt(self, template: str, **kwargs) -> str:
        return template.format(**kwargs)
