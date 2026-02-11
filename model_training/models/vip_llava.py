import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq

class VIPLLaVAMedCLM(nn.Module):
    def __init__(self, model_name: str, dtype: str = "float16", 
                 low_cpu_mem_usage: bool = True, device_map: str = "auto"):
        super().__init__()
        
        self.model_name = model_name
        
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map
        )
    
    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch.get('pixel_values')
        labels = batch.get('labels')
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )
        
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
    
    def save_pretrained(self, save_dir: str):
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
    
    def from_pretrained(self, load_dir: str):
        self.model = AutoModelForVision2Seq.from_pretrained(load_dir)
        self.processor = AutoProcessor.from_pretrained(load_dir)
