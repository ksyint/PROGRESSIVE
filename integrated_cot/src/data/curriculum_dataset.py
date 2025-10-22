import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import re



class CurriculumDataset(Dataset):

    def __init__(self, json_path):

        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.domain_map = {}

        for idx, item in enumerate(self.data):

            lesion_class = item["lesion_class"]
            image_path = item['image_path'] 
            
            modality = item['modality'] 
            domain = f"{lesion_class}_{modality}"  # create a domain key (e.g., "nodule_xray")
            
            if domain not in self.domain_map:
                self.domain_map[domain] = []
            self.domain_map[domain].append(idx) # map domains to sample indices
    
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        item = self.data[idx]

        image_path = item['image_path']  
        image_path2 = item['image_path2']  
        image_path3 = item['image_path3']  

        bbox=item['bbox']

        cot= item['cot']
        cot2= item['cot2']

        answer=item['answer']  
        answer2=item['answer2']  
        answer3=item['answer3']

        question=item["question"]
        question2=item["question2"]
        question3=item["question3"]

        lesion_class = item['lesion_class']
        modality = item["modality"]
        domain = f"{lesion_class}_{modality}"
        
        organ = item["organ"]
        rationale = f"{lesion_class}_{organ}"
        
        return {

            'image_path':image_path,
            'image_path2':image_path2,
            'image_path3':image_path3,

            'bbox':bbox,

            'cot':cot,
            'cot2':cot2,

            "answer":answer,
            "answer2":answer2,
            "answer3":answer3,

            "question":question,
            "question2":question2,
            "question3":question3,

            'domains': domain,
            'rationale':rationale
            }
    
    def get_domains(self):
        return list(self.domain_map.keys())
    
    def get_domain_indices(self, domain):
        return self.domain_map.get(domain, [])
    
    def set_stage(self, stage):
        self.stage = stage


        
def collate_fn(batch, processor, stage="easy", device="cuda"):
    
    conversations = []

    images = []
    bboxes = []
    domains = []
    rationales = []

    for item in batch:
        
        if stage == "easy":
            image_path = item['image_path']  # image with bbox visualization
            bbox = item['bbox']
            cot = item["cot"]
            answer = item['answer']
            question = item['question']
        elif stage == "medium":
            image_path = item['image_path2']  # same image, no bbox visualization
            bbox = item['bbox']
            cot = item["cot2"]
            answer = item['answer2']
            question = item['question2']
        elif stage == "hard":
            image_path = item['image_path3']  # different image, no cot, but same domain (lesion + modality)
            bbox = ""
            cot = ""                          
            answer = item['answer3']
            question = item['question3']

        
        
        image = Image.open(image_path).convert('RGB')
        domain = item['domains']
        rationale=item['rationale']
        
        images.append(image)
        bboxes.append(bbox)
        domains.append(domain)
        rationales.append(rationale)
        
        if stage == "hard":
            assistant_content = f"Answer: {answer}"
        else:
            assistant_content = f"Reasoning: {cot}\n\nAnswer: {answer}"
            
        conversation = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': question}
                ]
            },
            {
                'role': 'assistant',
                'content': [
                    {'type': 'text', 'text': assistant_content}
                ]
            }
        ]
        conversations.append(conversation)


    
    prompts = []
    for conv in conversations:
        prompt = processor.apply_chat_template([conv[0]], add_generation_prompt=True)
        prompts.append(prompt)
    
    inputs = processor(
        text=prompts,
        images=images,
        padding=True,
        return_tensors="pt"
    ).to(device)
    


    labels = inputs.input_ids.clone()
    for i, conv in enumerate(conversations):
        assistant_content = conv[1]['content'][0]['text']        
        assistant_tokens = processor.tokenizer.encode(assistant_content, add_special_tokens=False)
        input_ids_list = inputs.input_ids[i].tolist()
        
        assistant_start = -1
        # find the start of the assistant's response in the tokenized input
        for k in range(len(input_ids_list) - len(assistant_tokens) + 1):
            if input_ids_list[k:k+len(assistant_tokens)] == assistant_tokens:
                assistant_start = k
                break
        
        if assistant_start >= 0:
            labels[i, :assistant_start] = -100 # mask user prompt tokens (loss is not computed where the value is -100, by using ignore function)
        else:
            response_length = len(assistant_tokens)
            if response_length > 0 and len(input_ids_list) >= response_length:
                labels[i, :-response_length] = -100
            else:
                labels[i, :] = -100
                
    labels[labels == processor.tokenizer.pad_token_id] = -100 # mask padding tokens

    inputs['labels'] = labels
    inputs['bboxes'] = bboxes
    inputs['domains'] = domains
    inputs['stage'] = stage
    
    if stage!="hard":
        inputs["rationales"] = rationales
    
    return inputs