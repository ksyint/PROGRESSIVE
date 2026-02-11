from PIL import Image
import torch

def collate_fn(batch, processor, stage="easy", device="cuda"):
    conversations = []
    images = []
    bboxes = []
    domains = []
    rationales = []
    
    for item in batch:
        if stage == "easy":
            image_path = item['image_path']
            bbox = item['bbox']
            cot = item["cot"]
            answer = item['answer']
            question = item['question']
        elif stage == "medium":
            image_path = item['image_path2']
            bbox = item['bbox']
            cot = item["cot2"]
            answer = item['answer2']
            question = item['question2']
        elif stage == "hard":
            image_path = item['image_path3']
            bbox = ""
            cot = ""                          
            answer = item['answer3']
            question = item['question3']
        
        image = Image.open(image_path).convert('RGB')
        domain = item['domains']
        rationale = item['rationale']
        
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
    
    texts = processor.apply_chat_template(conversations, tokenize=False)
    
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    input_ids = inputs['input_ids']
    labels = input_ids.clone()
    
    for i, conv in enumerate(conversations):
        user_text = processor.apply_chat_template([conv[0]], tokenize=False)
        user_tokens = processor.tokenizer(user_text, add_special_tokens=False)['input_ids']
        labels[i, :len(user_tokens)] = -100
    
    inputs['labels'] = labels
    inputs['domains'] = domains
    inputs['bboxes'] = bboxes
    inputs['rationales'] = rationales
    
    return inputs
