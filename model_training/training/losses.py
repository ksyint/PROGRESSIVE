import torch
import torch.nn.functional as F

def compute_easy_medium_loss(outputs, batch, config, tokenizer):
    logits = outputs.logits
    labels = batch['labels']
    
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    per_token_loss = per_token_loss.view(shift_labels.size())
    
    total_loss = per_token_loss.sum() / (shift_labels != -100).sum()
    
    return {
        'loss': total_loss,
        'loss_ans': total_loss * 0.5,
        'loss_cot': total_loss * 0.5
    }

def compute_hard_loss(outputs, batch, config, tokenizer):
    logits = outputs.logits
    labels = batch['labels']
    
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return {
        'loss': loss,
        'loss_ans': loss
    }
