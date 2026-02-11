#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python model_training/train.py \
    --config configs/training.yaml
