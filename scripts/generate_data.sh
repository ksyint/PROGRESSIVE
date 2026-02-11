#!/bin/bash

python data_generation/main.py \
    --config configs/data_generation.yaml \
    --input data/input_detection.json \
    --output output/vqa_cot_data.json
