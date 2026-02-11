from collections import defaultdict
import numpy as np

class CurriculumScheduler:
    def __init__(self, config, domains):
        self.config = config
        self.domains = domains
        
        self.domain_ema_losses = {domain: float('inf') for domain in domains}
        self.ema_alpha = config.get('curriculum.ema_alpha', 0.9)
        
        self.easy_threshold = config.get('curriculum.easy_threshold', 2.0)
        self.medium_threshold = config.get('curriculum.medium_threshold', 1.5)
        
        self.warmup_epochs = config.get('curriculum.warmup_epochs', 2)
    
    def update_domain_loss(self, domain, loss):
        if domain not in self.domain_ema_losses:
            self.domain_ema_losses[domain] = loss
        else:
            self.domain_ema_losses[domain] = (
                self.ema_alpha * self.domain_ema_losses[domain] + 
                (1 - self.ema_alpha) * loss
            )
    
    def assign_stage(self, domain, epoch):
        if epoch < self.warmup_epochs:
            return "easy"
        
        domain_loss = self.domain_ema_losses.get(domain, float('inf'))
        
        if domain_loss > self.easy_threshold:
            return "easy"
        elif domain_loss > self.medium_threshold:
            return "medium"
        else:
            return "hard"
    
    def get_stage_distribution(self):
        distribution = defaultdict(int)
        for domain in self.domains:
            stage = self.assign_stage(domain, 100)
            distribution[stage] += 1
        return distribution
