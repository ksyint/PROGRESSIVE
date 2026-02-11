import torch
import os
import json
from tqdm import tqdm
from collections import defaultdict
import random
from .losses import compute_easy_medium_loss, compute_hard_loss
from .scheduler import CurriculumScheduler
from ..data import collate_fn

class Trainer:
    def __init__(self, model, train_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        
        self.device = torch.device("cuda")  
        self.setup_logging()
        
        self.optimizer = self.setup_optimizer()
        self.lr_scheduler = self.setup_lr_scheduler()
        
        domains = train_dataset.get_domains()
        self.curriculum_scheduler = CurriculumScheduler(config.to_dict() if hasattr(config, 'to_dict') else config.config, domains)
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.global_step = 0
    
    def setup_logging(self):
        from ..core.logger import TrainingLogger
        log_dir = self.config.get('logging.log_dir', 'logs')
        self.logger = TrainingLogger(log_dir)
    
    def setup_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('training.learning_rate', 1e-4),
            weight_decay=self.config.get('training.weight_decay', 0.01),
            betas=(self.config.get('training.beta1', 0.9), 
                   self.config.get('training.beta2', 0.999))
        )
    
    def setup_lr_scheduler(self):
        total_steps = len(self.train_dataset) * self.config.get('training.epochs', 10) // self.config.get('training.batch_size', 4)
        warmup_steps = int(total_steps * self.config.get('training.warmup_ratio', 0.1))
        
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.get('training.min_lr', 1e-6)
        )
    
    def create_domain_batches(self):
        batch_size = self.config.get('training.batch_size', 4)
        domain_samples = defaultdict(list)
        
        for idx in range(len(self.train_dataset)):
            item = self.train_dataset[idx]
            domain = item['domains']
            domain_samples[domain].append(item)
        
        for domain in domain_samples:
            random.shuffle(domain_samples[domain])
        
        batches = []
        for domain, samples in domain_samples.items():
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                batches.append(batch)
        
        random.shuffle(batches)
        return batches
    
    def create_stage_batch(self, items, stage):
        return collate_fn(
            items,
            processor=self.model.processor if not hasattr(self.model, 'module') else self.model.module.processor,
            stage=stage,
            device=self.device
        )
    
    def train_epoch(self, epoch):
        self.model.train()
        
        epoch_losses = defaultdict(lambda: defaultdict(list))
        grad_accum_steps = self.config.get('training.gradient_accumulation_steps', 1)
        
        domain_batches = self.create_domain_batches()
        
        progress_bar = tqdm(domain_batches, desc=f"Epoch {epoch}")
        
        for step, domain_batch in enumerate(progress_bar):
            staged_batch = []
            for item in domain_batch:
                domain = item['domains']
                stage = self.curriculum_scheduler.assign_stage(domain, epoch)
                item['assigned_stage'] = stage
                staged_batch.append(item)
            
            stage_groups = defaultdict(list)
            for item in staged_batch:
                stage_groups[item['assigned_stage']].append(item)
            
            total_loss = 0
            step_loss_dict = {}
            
            for stage, items in stage_groups.items():
                if not items:
                    continue
                
                batch = self.create_stage_batch(items, stage)
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    
                    if stage in ["easy", "medium"]:
                        loss_dict = compute_easy_medium_loss(
                            outputs, batch, self.config, self.model.processor.tokenizer
                        )
                    else:  
                        loss_dict = compute_hard_loss(outputs, batch, self.config, self.model.processor.tokenizer)
                    
                    loss = loss_dict['loss'] / grad_accum_steps
                    total_loss += loss
                
                for domain in batch['domains']:
                    for k, v in loss_dict.items():
                        if isinstance(v, torch.Tensor):
                            epoch_losses[domain][f"{stage}_{k}"].append(v.item())
                
                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        step_loss_dict[f"{stage}_{k}"] = v.item()
            
            if total_loss > 0:
                self.scaler.scale(total_loss).backward()
            
            if (step + 1) % grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get('training.max_grad_norm', 1.0)
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                self.global_step += 1
            
            progress_bar.set_postfix({
                'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        epoch_losses_avg = {}
        for domain, losses in epoch_losses.items():
            epoch_losses_avg[domain] = {
                k: sum(v) / len(v) if len(v) > 0 else 0.0 
                for k, v in losses.items()
            }
        
        return epoch_losses_avg
    
    def train(self):
        epochs = self.config.get('training.epochs', 10)
        output_dir = self.config.get('training.output_dir', './output')
        
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            epoch_losses = self.train_epoch(epoch)
            
            for domain, losses in epoch_losses.items():
                if f"easy_loss" in losses:
                    self.curriculum_scheduler.update_domain_loss(domain, losses['easy_loss'])
                elif f"medium_loss" in losses:
                    self.curriculum_scheduler.update_domain_loss(domain, losses['medium_loss'])
                elif f"hard_loss" in losses:
                    self.curriculum_scheduler.update_domain_loss(domain, losses['hard_loss'])
            
            if (epoch + 1) % self.config.get('training.save_every', 5) == 0:
                save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
                self.model.save_pretrained(save_path)
                self.logger.info(f"Saved checkpoint to {save_path}")
        
        final_path = os.path.join(output_dir, "final_model")
        self.model.save_pretrained(final_path)
        self.logger.info(f"Training complete! Model saved to {final_path}")
