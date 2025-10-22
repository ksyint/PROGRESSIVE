# MedCLM Automated CoT Data Generation Pipeline

This repository implements the automated data generation pipeline described in **Section 3.1** of the paper: *"MedCLM: Learning to Localize and Reason via a CoT-Curriculum in Medical Vision-Language Models"*.

The goal is to convert standard medical **detection** datasets (images + lesion bounding boxes + class labels) into rich **Visual Question Answering (VQA)** datasets suitable for training models like MedCLM. This pipeline leverages anatomical context (organ segmentation masks) and a powerful medical Vision-Language Model (HuatuoGPT-Vision 34B) to automatically generate question-answer pairs enriched with Chain-of-Thought (CoT) reasoning.

---

## Pipeline Overview

1. **Input:** A medical image with lesion annotations (bounding box + class label) and corresponding organ segmentation masks.

2. **Anatomical Contextualization:** For each lesion bounding box, the corresponding **host organ** is identified by calculating the IoU with organ masks.

3. **Seed Rationale Generation:** A simple factual sentence is created, linking the lesion and its host organ (e.g., "There is a nodule in the lung.").

4. **CoT-VQA Generation:** The medical VLM (HuatuoGPT-Vision 34B) is prompted with the image, the seed rationale, lesion class, organ name, and bounding box coordinates. Specific prompts are used to generate outputs corresponding to the **Easy, Medium Stages** defined in the MedCLM training curriculum.

5. **Output:** A JSON file where each entry contains the image paths, bounding box, generated questions, CoT rationales, answers, and relevant metadata (lesion class, organ, modality).

---


### Installation


```bash
git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision
mv HuatuoGPT-Vision Huatuo
pip install torch opencv-python transformers pyyaml numpy Pillow
```

## Data Preparation


### 1. Input Detection JSON (`data.input_json`)

A JSON file containing a list of dictionaries, where each dictionary represents an image and its annotations.
Prepare anatomy mask directory with process_detect_data.py

**Format:**

```json
[
  {
    "image_path": "/path/to/your/image1.png",
    "anatomy_path": "/path/to/image1/masks/",
    "annotations": [
      {
        "bbox": [150, 200, 350, 400],
        "class": "nodule"
      },
      {
        "bbox": [50, 80, 120, 150],
        "class": "opacity"
      }
    ]
  },
  {
    "image_path": "/path/to/image2.jpg",
    "anatomy_path": "/path/to/image2/masks/",
    "annotations": [
      {
        "bbox": [200, 250, 280, 320],
        "class": "mass"
      }
    ]
  }
]
```
**Field Descriptions:**

- `image_path`: Path to the medical image file.
- `anatomy_path`: Path to a directory containing segmentation masks for different organs relevant to the image. Mask filenames should correspond to organ names (e.g., `lung.png`, `liver.png`). Masks should be grayscale images where pixels belonging to the organ are non-zero.
- `annotations`: A list of lesions found in the image.
  - `bbox`: Pixel coordinates `[xmin, ymin, xmax, ymax]` of the lesion bounding box. The script will normalize these for internal calculations and output.
  - `class`: The type of lesion (e.g., "nodule", "mass", "pneumonia").

### Organ Segmentation Masks

The directory specified by `anatomy_path` for each image should contain image files (e.g., PNG, JPG) representing binary or grayscale masks for relevant organs. 

- The filename is used as the organ name (e.g., `left_lung.png` → organ name "left_lung").
- The script `utils.py` handles loading, resizing, and calculating bounding boxes from these masks.

### 2. Output Data

The script generates a JSON file (`output/vqa_cot_data.json`) suitable for training the MedCLM model. Each entry corresponds to one lesion annotation from the input and contains data structured for the three curriculum stages.

**Format:**
```json
[
  {
    "image_path": "/path/to/your/image1.png",
    "bbox": "[0.2500, 0.3333, 0.5833, 0.6667]",
    "question": "Generated question for easy stage?",
    "cot": "Generated CoT for easy stage including [0.2500, 0.3333, 0.5833, 0.6667]",
    "answer": "Generated answer for easy stage",
    
    "image_path2": "/path/to/your/image2.png",
    "question2": "Generated question for medium stage?",
    "cot2": "Generated CoT for medium stage including [0.2500, 0.3333, 0.5833, 0.6667]",
    "answer2": "Generated answer for medium stage",
    
    "lesion_class": "nodule",
    "organ": "lung",
    "modality": "xray"
  }
]

```
### 3. Key Output Fields

#### Easy Stage Data
- `image_path`: Image path (with bbox visualization)
- `bbox`: Normalized bounding box coordinates `[x1, y1, x2, y2]` as a string
- `question`: Generated question for easy stage
- `cot`: Generated Chain-of-Thought reasoning for easy stage (includes bbox)
- `answer`: Generated answer for easy stage

#### Medium Stage Data
- `image_path2`: Image path (without bbox visualization)
- `question2`: Generated question for medium stage
- `cot2`: Generated Chain-of-Thought reasoning for medium stage (includes bbox)
- `answer2`: Generated answer for medium stage


#### Metadata
- `lesion_class`: Original lesion class from input
- `organ`: Determined host organ via IoU calculation
- `modality`: Inferred from image_path (e.g., "xray", "ct", "mri")
