import os
import argparse
import yaml
import torch
import random
import numpy as np
from src import VIPLLaVAMedCLM, CurriculumDataset, Trainer


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
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    
    set_seed(config['training']['seed'])
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    


    print("Initializing model...")
    model = VIPLLaVAMedCLM(
        model_name=config['model']['model_name'],
        dtype=config['model']['dtype'],
        low_cpu_mem_usage=config['model']['low_cpu_mem_usage'],
        device_map="auto"
    )
    

    print("Loading datasets...")
    train_dataset = CurriculumDataset(
        json_path=config['data']['train_json']
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