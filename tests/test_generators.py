import unittest
from data_generation.generators import EasyGenerator, MediumGenerator, HardGenerator

class MockModel:
    def inference(self, query, image_path):
        return "Question: Test question | Reasoning: Test reasoning | Answer: Test answer"

class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.easy_gen = EasyGenerator(self.model)
        self.medium_gen = MediumGenerator(self.model)
        self.hard_gen = HardGenerator(self.model)
    
    def test_easy_generator(self):
        result = self.easy_gen.generate(
            "test.jpg", "seed", "nodule", "lung", "[0.1, 0.2, 0.3, 0.4]"
        )
        self.assertIn('question', result)
        self.assertIn('cot', result)
        self.assertIn('answer', result)
    
    def test_medium_generator(self):
        result = self.medium_gen.generate(
            "test.jpg", "seed", "nodule", "lung", "[0.1, 0.2, 0.3, 0.4]"
        )
        self.assertIn('question2', result)
        self.assertIn('cot2', result)
        self.assertIn('answer2', result)
    
    def test_hard_generator(self):
        result = self.hard_gen.generate(
            "test.jpg", "seed", "nodule", "lung"
        )
        self.assertIn('question3', result)
        self.assertIn('answer3', result)

if __name__ == '__main__':
    unittest.main()
