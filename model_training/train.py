import os
import argparse
import yaml
import torch
import random
import numpy as np
from core import TrainingConfig
from models import VIPLLaVAMedCLM
from data import CurriculumDataset
from training import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='MedCLM Training')
    parser.add_argument('--config', type=str, default='configs/training.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    config_dict = load_config(args.config)
    config = TrainingConfig(args.config)
    
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("Initializing model...")
    model = VIPLLaVAMedCLM(
        model_name=config.model_name,
        dtype=config_dict['model'].get('dtype', 'float16'),
        low_cpu_mem_usage=config_dict['model'].get('low_cpu_mem_usage', True),
        device_map="auto"
    )
    
    print("Loading datasets...")
    train_dataset = CurriculumDataset(
        json_path=config.train_json
    )
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        config=config,
    )
    
    print("Starting training...")
    trainer.train()
    print("Training completed!")

if __name__ == '__main__':
    main()
