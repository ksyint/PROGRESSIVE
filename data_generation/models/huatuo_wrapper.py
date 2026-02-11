import sys
import os
from typing import List

class ModelInferenceError(Exception):
    pass

class HuatuoWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._load_model()
    
    def _load_model(self):
        try:
            huatuo_dir = os.path.dirname(self.model_path)
            huatuo_path = os.path.join(huatuo_dir, 'Huatuo')
            
            if not os.path.exists(huatuo_path):
                huatuo_path = os.path.join(os.getcwd(), 'Huatuo')
            
            if huatuo_path not in sys.path:
                sys.path.insert(0, huatuo_path)
            
            from Huatuo.cli import HuatuoChatbot
            self._model = HuatuoChatbot(self.model_path)
        except Exception as e:
            raise ModelInferenceError(f"Failed to load Huatuo model: {e}")
    
    def inference(self, query: str, image_path: str) -> str:
        try:
            image_path_list = [image_path]
            output = self._model.inference(query, image_path_list)
            return output
        except Exception as e:
            raise ModelInferenceError(f"Inference failed: {e}")
    
    def batch_inference(self, queries: List[str], image_paths: List[str]) -> List[str]:
        results = []
        for query, image_path in zip(queries, image_paths):
            try:
                result = self.inference(query, image_path)
                results.append(result)
            except ModelInferenceError as e:
                print(f"Skipping failed inference: {e}")
                results.append("")
        return results
