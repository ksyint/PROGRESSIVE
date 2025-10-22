import torch
import numpy as np
from PIL import Image
import shutil
import os
from cxas import CXAS

# make anatomy segmentation masks

input_image_dir="/path/dir"
out_dir="/path/dir2"

os.makedirs(out_dir,exist_ok=True)
model = CXAS(
    model_name = 'UNet_ResNet50_default',
    gpus       = '0'
)

for img in os.listdir(input_image_dir):
    
    image_path=os.path.join(input_image_dir, img)
    _ = model.process_file(
            filename = image_path,
            do_store = True, 
            output_directory = out_dir,
            storage_type = 'jpg',
            )
