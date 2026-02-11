from .base_generator import BaseGenerator

HARD_TEMPLATE = """Based on similar medical cases showing {seed}, generate a general diagnostic question.
Focus on clinical reasoning without specific location details.

Format exactly as: Question: [question] | Answer: [answer]"""

HARD_KEYS = ["question3", "answer3"]

class HardGenerator(BaseGenerator):
    def generate(self, image_path: str, seed: str, lesion_class: str, 
                organ: str, bbox_str: str = "") -> dict:
        
        prompt = self.create_prompt(
            HARD_TEMPLATE,
            seed=seed
        )
        
        response = self.model.inference(prompt, image_path)
        
        parsed = self.parse_response(response, HARD_KEYS)
        
        return parsed
