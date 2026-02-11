from collections import defaultdict
from typing import Dict
import numpy as np

class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def get_average(self, key: str, last_n: int = None) -> float:
        values = self.metrics[key]
        
        if not values:
            return 0.0
        
        if last_n:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def get_latest(self, key: str) -> float:
        if not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]
    
    def reset(self):
        self.metrics.clear()
    
    def get_summary(self) -> Dict[str, float]:
        summary = {}
        for key in self.metrics:
            summary[f"{key}_mean"] = self.get_average(key)
            summary[f"{key}_last"] = self.get_latest(key)
        return summary
