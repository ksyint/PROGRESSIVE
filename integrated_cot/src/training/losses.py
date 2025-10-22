import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torchvision.transforms as T
from .attn_loss import compute_attention_mask_loss, compute_grounding_loss, box_loss
from .utils import get_token_positions, parse_response_text



def compute_answer_loss(outputs, labels, tokenizer, batch_size):
    device = outputs.logits.device
    # create a mask to calculate loss only on the 'answer' part
    answer_mask = torch.zeros(labels.shape, dtype=torch.float, device=device)
    
    for i in range(batch_size):

        label_ids = labels[i][labels[i] != -100].cpu().tolist()
        parsed, text = parse_response_text(tokenizer, label_ids)
        
        if parsed['answer_start'] is not None:
            # find token positions for the answer characters

            token_start, token_end = get_token_positions(
                tokenizer, text, parsed['answer_start'], parsed['answer_end']
            )
            
            if token_start is not None and token_end is not None:
                max_len = labels.shape[1]
                token_end = min(token_end, max_len)
                answer_mask[i, token_start:token_end] = 1 # unmask only answer tokens
    

    
    masked_labels = labels.clone().to(device)
    masked_labels[answer_mask == 0] = -100 # mask all non-answer tokens
    
    logits = outputs.logits
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = masked_labels[..., 1:].contiguous()
    
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss


def compute_cot_loss(outputs, labels, tokenizer, batch_size):
    
    device = outputs.logits.device
    # create a mask to calculate loss only on the 'reasoning' (cot) part
    cot_mask = torch.zeros(labels.shape, dtype=torch.float, device=device)
    
    for i in range(batch_size):
        label_ids = labels[i][labels[i] != -100].cpu().tolist()
        parsed, text = parse_response_text(tokenizer, label_ids)
        
        if parsed['reasoning_start'] is not None and parsed['reasoning_end'] is not None:
            # find token positions for the reasoning characters
            token_start, token_end = get_token_positions(
                tokenizer, text, parsed['reasoning_start'], parsed['reasoning_end']
            )
            
            if token_start is not None and token_end is not None:
                max_len = labels.shape[1]
                token_end = min(token_end, max_len)
                cot_mask[i, token_start:token_end] = 1

    
    masked_labels = labels.clone().to(device)
    masked_labels[cot_mask == 0] = -100 # mask all non-cot tokens
    
    logits = outputs.logits
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = masked_labels[..., 1:].contiguous()
    
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss


def compute_easy_medium_loss(outputs, batch, config, tokenizer):
   
    lambda_ans = config['losses']['lambda_ans']
    lambda_cot = config['losses']['lambda_cot']
    
    labels = batch['labels']
    batch_size = labels.shape[0]
    
    loss_ans = compute_answer_loss(outputs, labels, tokenizer, batch_size)
    loss_cot = compute_cot_loss(outputs, labels, tokenizer, batch_size)
    
    total_loss = (lambda_ans * loss_ans + 
                  lambda_cot * loss_cot )
    
    loss_dict = {
        'loss': total_loss,
    }

    return loss_dict


def compute_hard_loss(outputs, batch, config, tokenizer):
    
    labels = batch['labels']
    batch_size = labels.shape[0]
    
    loss_ans = compute_answer_loss(outputs, labels, tokenizer, batch_size)
    
    total_loss = loss_ans
    
    loss_dict = {
        'loss': total_loss,
    }
    
    return loss_dict
