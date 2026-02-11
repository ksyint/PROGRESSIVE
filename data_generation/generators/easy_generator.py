from .base_generator import BaseGenerator

EASY_TEMPLATE = """Given this medical image showing: {seed}.
The {lesion_class} is suspected within the box region {bbox_str} which is [xmin, ymin, xmax, ymax].
(xmin, ymin) is top-left, (xmax, ymax) is bottom-right.

Generate a specific diagnostic question targeting the highlighted region.
Provide: 1) The question, 2) Step-by-step reasoning including the box {bbox_str}, 3) A concise final answer.

Format exactly as: Question: [question] | Reasoning: [reasoning including {bbox_str}] | Answer: [answer]"""

EASY_KEYS = ["question", "cot", "answer"]

class EasyGenerator(BaseGenerator):
    def generate(self, image_path: str, seed: str, lesion_class: str, 
                organ: str, bbox_str: str) -> dict:
        
        prompt = self.create_prompt(
            EASY_TEMPLATE,
            seed=seed,
            lesion_class=lesion_class,
            bbox_str=bbox_str
        )
        
        response = self.model.inference(prompt, image_path)
        
        parsed = self.parse_response(response, EASY_KEYS)
        
        return parsed
