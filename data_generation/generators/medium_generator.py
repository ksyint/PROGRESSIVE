from .base_generator import BaseGenerator

MEDIUM_TEMPLATE = """Analyze the entire medical image, noting the finding: {seed}.
The {lesion_class} is located in the {organ} around region {bbox_str} [xmin, ymin, xmax, ymax].

Generate a question about identifying abnormalities in the whole image.
Provide: 1) The question, 2) Reasoning describing the finding's location (mentioning {bbox_str}), 3) A final answer including location context.

Format exactly as: Question: [question] | Reasoning: [reasoning including {bbox_str}] | Answer: [answer]"""

MEDIUM_KEYS = ["question2", "cot2", "answer2"]

class MediumGenerator(BaseGenerator):
    def generate(self, image_path: str, seed: str, lesion_class: str, 
                organ: str, bbox_str: str) -> dict:
        
        prompt = self.create_prompt(
            MEDIUM_TEMPLATE,
            seed=seed,
            lesion_class=lesion_class,
            organ=organ,
            bbox_str=bbox_str
        )
        
        response = self.model.inference(prompt, image_path)
        
        parsed = self.parse_response(response, MEDIUM_KEYS)
        
        return parsed
