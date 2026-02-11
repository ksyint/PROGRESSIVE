import json
import os
import argparse
from core.config import Config
from core.logger import get_logger
from pipeline import DataGenerationPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Medical VQA Data Generation')
    parser.add_argument('--config', type=str, default='configs/data_generation.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, default=None,
                       help='Input detection JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output VQA JSON file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = Config(args.config)
    logger = get_logger('Main')
    
    input_json = args.input or config.get('data.input_json')
    output_json = args.output or config.get('data.output_vqa')
    
    logger.info("Loading detection data")
    with open(input_json, 'r') as f:
        detection_data = json.load(f)
    
    logger.info("Initializing pipeline")
    pipeline = DataGenerationPipeline(config)
    
    logger.info("Processing data")
    results = pipeline.process_detection_data(detection_data)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    logger.info(f"Saving results to {output_json}")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Complete! Generated {len(results)} samples")

if __name__ == '__main__':
    main()
