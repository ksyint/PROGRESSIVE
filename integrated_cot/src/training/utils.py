import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torchvision.transforms as T



def parse_response_text(tokenizer, token_ids):
    # decodes tokens and uses regex to find reasoning and answer parts
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    reasoning_pattern = r'Reasoning:\s*'
    answer_pattern = r'\n\nAnswer:\s*'
    box_pattern = r'\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]'
    
    result = {
        'reasoning_start': None,
        'reasoning_end': None,
        'answer_start': None,
        'answer_end': None,
        'box': None
    }
    
    reasoning_match = re.search(reasoning_pattern, text)
    if reasoning_match:
        result['reasoning_start'] = reasoning_match.end() # cot ends where answer begins
        
    answer_match = re.search(answer_pattern, text)
    if answer_match:
        result['answer_start'] = answer_match.end()
        result['answer_end'] = len(text)
        
        if result['reasoning_start'] is not None:
            result['reasoning_end'] = answer_match.start()
    
    if result['reasoning_start'] is not None:
        # also parse bbox coordinates from the reasoning text
        reasoning_text = text[result['reasoning_start']:result['reasoning_end']]
        box_match = re.search(box_pattern, reasoning_text)
        if box_match:
            result['box'] = [float(box_match.group(i)) for i in range(1, 5)]
    
    return result, text


def get_token_positions(tokenizer, text, char_start, char_end):
    # finds the token indices corresponding to character start/end positions
    if char_start is None or char_end is None:
        return None, None
    
    full_tokens = tokenizer.encode(text, add_special_tokens=False)
    
    token_start_idx = None
    token_end_idx = None
    current_pos = 0
    
    for i, token_id in enumerate(full_tokens):
        token_text = tokenizer.decode([token_id])
        token_len = len(token_text)
        
        if token_start_idx is None and current_pos >= char_start:
            token_start_idx = i
        
        if current_pos + token_len >= char_end:
            token_end_idx = i + 1
            break
            
        current_pos += token_len
    
    return token_start_idx, token_end_idx

