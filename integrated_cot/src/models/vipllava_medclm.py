import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VipLlavaForConditionalGeneration, AutoProcessor
import cv2
import numpy as np



class VIPLLaVAMedCLM(nn.Module):
    
    def __init__(self, model_name, dtype="bfloat16", low_cpu_mem_usage=True, device_map="auto"):
        
        super().__init__()
        
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        self.model = VipLlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "right"

        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
                
        self.dtype = torch_dtype

    
    def forward(self, batch):
        
        model_inputs = {
            k: v for k, v in batch.items() 
            if k in ['input_ids', 'attention_mask', 'pixel_values', 'labels']
        }
        
        outputs = self.model(
            **model_inputs,
            output_attentions=True, 
            output_hidden_states=True, 
            return_dict=True
        )
        
        if hasattr(outputs, 'vision_hidden_states'):
            # extract the vision features 
            model_config = self.model.config
            feature_layer_index = model_config.vision_feature_layer 
            selected_features = outputs.vision_hidden_states[feature_layer_index]
            # skip the [cls] token (index 0) and keep only patch features
            outputs.image_features = selected_features[:, 1:, :] 

        return outputs
    
    def generate(self, inputs, max_length=512, temperature=0.7, top_p=0.9):
  
        model_inputs = {
            k: v for k, v in inputs.items() 
            if k in ['input_ids', 'attention_mask', 'pixel_values']
        }
        
        outputs = self.model.generate(
            **model_inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        return outputs