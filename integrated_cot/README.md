# MedCLM: Learning to Localize and Reason via a CoT-Curriculum in Medical VLMs

This repository contains the implementation for the paper "MedCLM: Learning to Localize and Reason via a CoT-Curriculum in Medical Vision-Language Models".

MedCLM introduces an **Integrated CoT–Curriculum Strategy** for fine-tuning medical Vision-Language Models (VLMs) like VIP-LLaVA. The goal is to improve localization and reasoning capabilities by progressively training the model through stages of increasing difficulty.

## Key Features

* **Integrated CoT-Curriculum:** A three-stage training approach:
    * 🔵 **Easy Stage:** Explicit localization using images with bounding box visualizations and Chain-of-Thought (CoT) rationales.
    * 🟢 **Medium Stage:** Implicit localization using the same images *without* box visualizations. Still uses CoT rationales.
    * 🔴 **Hard Stage:** Weakly-supervised reasoning using different, but related, images and only supervising the final answer ($L_{\text{ans}}$), without CoT.
* **Domain-Aware Curriculum Scheduler:** Adaptively assigns samples to Easy/Medium/Hard stages based on per-domain (lesion type + modality) difficulty tracking (EMA of losses) and overall training progress (loss plateaus, gaps).
* **Stage-Specific Losses:** Utilizes different combinations of losses ($L_{\text{ans}}$, $L_{\text{cot}}$ depending on the training stage.

## Dataset Preparation

The training script expects a JSON file specified by `data/data.train_json` in the configuration. Each entry in the JSON list should follow this structure:

```json
[
  {
    ========== Easy Stage Data ==========
    "image_path": "/path/to/image_with_bbox_visualization.jpg", Image with bbox drawn
    "bbox": "[0.1, 0.2, 0.3, 0.4]", // related to easy, medium
    "question": "Question related to the bbox region?",
    "cot": "Reasoning text... MUST include bbox coordinates like [0.1, 0.2, 0.3, 0.4]", CRITICAL: Bbox coords should be in CoT
    "answer": "Final short answer for easy stage",

    ========== Medium Stage Data ==========
    "image_path2": "/path/to/same_image_without_bbox.jpg", Same image as above, no bbox drawn
    "question2": "Question about the whole image?",
    "cot2": "Reasoning text... MUST include bbox coordinates like [0.1, 0.2, 0.3, 0.4]", CRITICAL: Bbox coords should be in CoT
    "answer2": "Final short answer for medium stage",

    ========== Hard Stage Data ==========
    "image_path3": "/path/to/different_similar_case.jpg", Different image, same domain (lesion_modality)
    "question3": "General question about findings?",
    "answer3": "Final short answer for hard stage",

    ========== Metadata ==========
    "lesion_class": "nodule", related to easy, medium, hard
    "organ": "lung", related to easy, medium
    "modality": "xray" related to easy, medium, hard
  },

  {
   more samples
  },

   

]
