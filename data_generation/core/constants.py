MODALITIES = ["xray", "ct", "mri", "ultrasound"]

DEFAULT_MASK_THRESHOLD = 127

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
MASK_EXTENSIONS = [".png", ".jpg", ".jpeg"]

ORGAN_NAMES = [
    "lung", "liver", "heart", "kidney", "spleen", 
    "pancreas", "stomach", "intestine", "bladder"
]

LESION_CLASSES = [
    "nodule", "mass", "opacity", "infiltrate", 
    "pneumonia", "consolidation", "effusion"
]

BBOX_FORMAT_PIXEL = "pixel"
BBOX_FORMAT_NORMALIZED = "normalized"

MIN_BBOX_SIZE = 10
MAX_BBOX_SIZE = 1000

DEFAULT_IMAGE_SIZE = (512, 512)

GENERATION_STAGES = ["easy", "medium", "hard"]

EASY_STAGE = "easy"
MEDIUM_STAGE = "medium"
HARD_STAGE = "hard"

VQA_MODE = "vqa"
CAPTION_MODE = "caption"

DEFAULT_MAX_LENGTH = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

PROMPT_TEMPLATES = {
    "easy": """Given this medical image showing: {seed}.
The {lesion_class} is suspected within the box region {bbox_str} which is [xmin, ymin, xmax, ymax].
(xmin, ymin) is top-left, (xmax, ymax) is bottom-right.

Generate a specific diagnostic question targeting the highlighted region.
Provide: 1) The question, 2) Step-by-step reasoning including the box {bbox_str}, 3) A concise final answer.

Format exactly as: Question: [question] | Reasoning: [reasoning including {bbox_str}] | Answer: [answer]""",
    
    "medium": """Analyze the entire medical image, noting the finding: {seed}.
The {lesion_class} is located in the {organ} around region {bbox_str} [xmin, ymin, xmax, ymax].

Generate a question about identifying abnormalities in the whole image.
Provide: 1) The question, 2) Reasoning describing the finding's location (mentioning {bbox_str}), 3) A final answer including location context.

Format exactly as: Question: [question] | Reasoning: [reasoning including {bbox_str}] | Answer: [answer]""",
    
    "hard": """Based on similar medical cases showing {seed}, generate a general diagnostic question.
Focus on clinical reasoning without specific location details.

Format exactly as: Question: [question] | Answer: [answer]"""
}

RESPONSE_KEYS = {
    "easy": ["question", "cot", "answer"],
    "medium": ["question2", "cot2", "answer2"],
    "hard": ["question3", "answer3"]
}
