
import json
import yaml
import os
from data_generator import MedCLMDataGenerator



def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():


    config = load_config()
    mode   = config["mode"]
    
    with open(config["data"]["input_json"], 'r') as f:
        detection_data = json.load(f)
    
    generator = MedCLMDataGenerator(config["model"]["huatuo_path"])
    
    results = generator.process_detection_data(detection_data, output_type=mode)
    
    output_path = config["data"]["output_vqa"] if mode == "vqa" else config["data"]["output_caption"]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(results)} {mode} samples")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
