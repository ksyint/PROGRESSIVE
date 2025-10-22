import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torchvision.transforms as T
from utils import parse_response_text, get_token_positions



def _parse_bbox_string(bbox_str_list, normalized, device, image_size=(224, 224)):
 
    bboxes = []
    for i, bbox_str in enumerate(bbox_str_list):
        if not bbox_str:
            bboxes.append([0, 0, 1, 1]) 
            continue
            
        try:

            coords = [float(c) for c in re.findall(r"([0-9.]+)", bbox_str)]        
            if not normalized:
    
                h, w = image_size
                x1, y1, x2, y2 = coords
                coords = [x1 / w, y1 / h, x2 / w, y2 / h]

            bboxes.append(coords)
        except Exception as e:
            bboxes.append([0, 0, 1, 1]) 

    return torch.tensor(bboxes, dtype=torch.float32, device=device)

def _get_patch_features_for_bbox(image_features, bboxes_norm, grid_size=(24, 24)):
   
    B, NumPatches, D = image_features.shape
    H_grid, W_grid = grid_size
    assert NumPatches == H_grid * W_grid

    patch_indices = torch.arange(NumPatches, device=image_features.device)
    i_indices = patch_indices // W_grid
    j_indices = patch_indices % W_grid

    patch_x1 = j_indices.float() / W_grid
    patch_y1 = i_indices.float() / H_grid
    patch_x2 = (j_indices + 1).float() / W_grid
    patch_y2 = (i_indices + 1).float() / H_grid

    patch_boxes = torch.stack([patch_x1, patch_y1, patch_x2, patch_y2], dim=1)
    
    pooled_features = []
    for i in range(B):
        gt_bbox = bboxes_norm[i].unsqueeze(0) 
   
        lt = torch.max(patch_boxes[:, :2], gt_bbox[:, :2]) 
        rb = torch.min(patch_boxes[:, 2:], gt_bbox[:, 2:])

        wh = (rb - lt).clamp(min=0) 
        overlap = wh[:, 0] * wh[:, 1] 
        
        overlapping_patch_indices = torch.where(overlap > 0)[0]
        
        if overlapping_patch_indices.numel() > 0:
            rB = image_features[i, overlapping_patch_indices, :].mean(dim=0)
        else:
            rB = image_features[i].mean(dim=0)
        
        pooled_features.append(rB)

    return torch.stack(pooled_features)


def compute_grounding_loss(model, outputs, batch, tokenizer):
   
    device = outputs.logits.device
    
   
    grid_size = (24, 24) 
    if outputs.image_features.shape[1] != grid_size[0] * grid_size[1]:
        num_patches = outputs.image_features.shape[1]
        grid_h = int(num_patches**0.5)
        grid_w = num_patches // grid_h
        grid_size = (grid_h, grid_w)

    bboxes_norm = _parse_bbox_string(batch['bboxes'], normalized=True, device=device)
    rB = _get_patch_features_for_bbox(outputs.image_features, bboxes_norm, grid_size=grid_size)

    anchor_texts = [d.replace('_', ' in ') for d in batch['rationales']] 
    anchor_tokens = tokenizer(
        anchor_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)
    
    text_embeddings = model.model.get_input_embeddings()

    t_l_o = text_embeddings(anchor_tokens.input_ids).mean(dim=1) 

    loss = 1.0 - F.cosine_similarity(rB, t_l_o).mean()
    
    return loss


def _create_soft_mask(bbox_str_list, target_size, device, kernel_size=9, sigma=3.0):
   
    B = len(bbox_str_list)
    H_grid, W_grid = target_size
    
    bboxes_norm = _parse_bbox_string(bbox_str_list, normalized=True, device=device)
    
    x1 = (bboxes_norm[:, 0] * W_grid).clamp(0, W_grid - 1).long()
    y1 = (bboxes_norm[:, 1] * H_grid).clamp(0, H_grid - 1).long()
    x2 = (bboxes_norm[:, 2] * W_grid).clamp(1, W_grid).long()
    y2 = (bboxes_norm[:, 3] * H_grid).clamp(1, H_grid).long()

    masks = torch.zeros(B, H_grid, W_grid, device=device)
    
    for i in range(B):
        if torch.any(bboxes_norm[i] != 0) or torch.all(bboxes_norm[i] != 1): 
            masks[i, y1[i]:y2[i], x1[i]:x2[i]] = 1.0

    blurrer = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma).to(device)
    soft_masks = blurrer(masks.unsqueeze(1)).squeeze(1) 
    
    mask_sum = soft_masks.view(B, -1).sum(dim=1) + 1e-8
    soft_masks = soft_masks.view(B, -1) / mask_sum.unsqueeze(1)
    
    return soft_masks.view(B, H_grid, W_grid) 


def compute_attention_mask_loss(outputs, batch, grid_size=(24, 24)):
  
    device = outputs.logits.device
    B = outputs.logits.shape[0]


    mB = _create_soft_mask(batch['bboxes'], target_size=grid_size, device=device)

    vision_attentions = outputs.vision_attentions[-1] 
    
    attn_map = vision_attentions.mean(dim=1)
    
    cls_to_patch_attn = attn_map[:, 0, 1:] 
    
    if cls_to_patch_attn.shape[1] != grid_size[0] * grid_size[1]:
        num_patches = cls_to_patch_attn.shape[1]
        grid_h = int(num_patches**0.5)
        grid_w = num_patches // grid_h
        grid_size_actual = (grid_h, grid_w)
        mB = _create_soft_mask(batch['bboxes'], target_size=grid_size_actual, device=device)
    else:
        grid_size_actual = grid_size
        
    cls_to_patch_attn = cls_to_patch_attn.view(B, *grid_size_actual) 

    attn_prob = F.softmax(cls_to_patch_attn.view(B, -1), dim=1) 
    
    
    attn_log_prob = (attn_prob + 1e-8).log().view(B, -1)
    mB_prob = mB.view(B, -1)

    loss = F.kl_div(attn_log_prob, mB_prob, reduction='batchmean')
    
    return loss

def extract_boxes_from_batch(tokenizer, outputs, labels, batch_size):
    
    pred_boxes = []
    gt_boxes = []
    
    pred_ids = torch.argmax(outputs.logits, dim=-1)
    for i in range(batch_size):
        pred_token_ids = pred_ids[i].cpu().tolist()
        parsed, _ = parse_response_text(tokenizer, pred_token_ids)
        pred_boxes.append(parsed['box'])
    
    for i in range(batch_size):
        label_ids = labels[i][labels[i] != -100].cpu().tolist()
        parsed, _ = parse_response_text(tokenizer, label_ids)
        gt_boxes.append(parsed['box'])
    
    return pred_boxes, gt_boxes


def compute_box_loss(pred_boxes, gt_boxes):
   
    valid_pairs = []
    
    for pred_box, gt_box in zip(pred_boxes, gt_boxes):
        if pred_box is not None and gt_box is not None:
            valid_pairs.append((pred_box, gt_box))
    
    if len(valid_pairs) == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    pred_tensor = torch.tensor([p[0] for p in valid_pairs], dtype=torch.float32)
    gt_tensor = torch.tensor([p[1] for p in valid_pairs], dtype=torch.float32)
    
    loss = F.l1_loss(pred_tensor, gt_tensor)
    
    return loss

def box_loss(tokenizer, outputs, labels, batch_size):

    #lambda_box = config['losses']['lambda_box']
    
    pred_boxes, gt_boxes = extract_boxes_from_batch(tokenizer, outputs, labels, batch_size)
    device = outputs.logits.device
    loss_box = compute_box_loss(pred_boxes, gt_boxes)
    loss_box = loss_box.to(device)

    return loss_box

