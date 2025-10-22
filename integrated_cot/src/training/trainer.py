import torch
import os
import json
import logging
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
        self.curriculum_scheduler = CurriculumScheduler(config, domains) # handles curriculum logic
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.global_step = 0
        
    def setup_logging(self):
        
        log_dir = self.config['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "train.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

    
    def setup_optimizer(self):

        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
    
    def setup_lr_scheduler(self):

        total_steps = len(self.train_dataset) * self.config['training']['epochs'] // self.config['training']['batch_size']
        warmup_steps = int(total_steps * self.config['training']['warmup_ratio'])
        
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config['training']['min_lr']
        )
    
    def create_domain_batches(self):
        
        # group samples by domain (lesion_modality) and shuffle

        batch_size = self.config['training']['batch_size']
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
        
        # process a list of items into a tensor batch for a specific stage

        return collate_fn(
            items,
            processor=self.model.processor if not hasattr(self.model, 'module') else self.model.module.processor,
            stage=stage,
            device=self.device
        )
    
    def train_epoch(self, epoch):

        self.model.train()
        
        epoch_losses = defaultdict(lambda: defaultdict(list))
        grad_accum_steps = self.config['training']['gradient_accumulation_steps']
        
        domain_batches = self.create_domain_batches()
        
        progress_bar = tqdm(domain_batches, desc=f"Epoch {epoch}")
        
        for step, domain_batch in enumerate(progress_bar):
            staged_batch = []
            for item in domain_batch:
                domain = item['domains']
                # ask the scheduler which stage (easy, medium, hard) to use
                stage = self.curriculum_scheduler.assign_stage(domain, epoch)
                item['assigned_stage'] = stage
                staged_batch.append(item)
            
            # group items by their assigned stage
            stage_groups = defaultdict(list)
            for item in staged_batch:
                stage_groups[item['assigned_stage']].append(item)
            
            total_loss = 0
            step_loss_dict = {}
            
            # process each stage group separately
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
                        # k : "loss" , v : 0.21 (ex)
                        # store losses by domain for the scheduler
                        if isinstance(v, torch.Tensor):
                            epoch_losses[domain][f"{stage}_{k}"].append(v.item())
                
                for k, v in loss_dict.items():
                    # store losses for logging
                    if isinstance(v, torch.Tensor):
                        step_loss_dict[f"{stage}_{k}"] = v.item()
            
            if total_loss > 0:
                self.scaler.scale(total_loss).backward()
            
            if (step + 1) % grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['max_grad_norm']
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                self.global_step += 1
            
            progress_bar.set_postfix({
                'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
                'lr': self.optimizer.param_groups[0]['lr'],
                **{k: f"{v:.4f}" for k, v in list(step_loss_dict.items())[:3]} 
            })
        
        epoch_losses_avg = {}
        for domain, losses in epoch_losses.items():
            # calculate average loss per domain for the scheduler
            epoch_losses_avg[domain] = {
                k: sum(v) / len(v) if len(v) > 0 else 0.0 
                for k, v in losses.items()
            }
        
        return epoch_losses_avg

    def save_checkpoint(self, epoch):
    
        
        checkpoint_dir = os.path.join(self.config['training']['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        save_path = os.path.join(checkpoint_dir, f'epoch_{epoch}')
        
        os.makedirs(save_path, exist_ok=True)
        
        model_to_save.model.save_pretrained(save_path)
        model_to_save.processor.save_pretrained(save_path)
        
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'curriculum_state': self.curriculum_scheduler.__dict__,
        }, os.path.join(save_path, 'training_state.pt'))
        
        self.logger.info(f"Saved checkpoint to {save_path}")
    
    def train(self):

        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.config['training']['epochs']}")
        self.logger.info(f"Batch size: {self.config['training']['batch_size']}")
        self.logger.info(f"Learning rate: {self.config['training']['learning_rate']}")
        
        for epoch in range(self.config['training']['epochs']):
            
            epoch_losses = self.train_epoch(epoch)
            
            if epoch_losses:  
                # update the curriculum scheduler based on epoch losses
                self.curriculum_scheduler.step(epoch, epoch_losses)
            
            if epoch % self.config['training']['save_every_epochs'] == 0:
                self.save_checkpoint(epoch)
        
        self.logger.info("Training completed!")
        