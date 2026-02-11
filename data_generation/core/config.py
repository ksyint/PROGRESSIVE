import yaml
import os
from typing import Any
from .exceptions import ConfigurationError

class Config:
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: str):
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def to_dict(self):
        return self.config.copy()
