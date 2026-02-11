import yaml
from typing import Any

class TrainingConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def model_name(self) -> str:
        return self.get('model.model_name')
    
    @property
    def train_json(self) -> str:
        return self.get('data.train_json')
    
    @property
    def batch_size(self) -> int:
        return self.get('training.batch_size', 4)
    
    @property
    def learning_rate(self) -> float:
        return self.get('training.learning_rate', 1e-4)
    
    @property
    def epochs(self) -> int:
        return self.get('training.epochs', 10)
    
    @property
    def output_dir(self) -> str:
        return self.get('training.output_dir', './output')
    
    @property
    def seed(self) -> int:
        return self.get('training.seed', 42)
