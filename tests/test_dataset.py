import unittest
import json
import tempfile
import os
from model_training.data import CurriculumDataset

class TestCurriculumDataset(unittest.TestCase):
    def setUp(self):
        self.test_data = [
            {
                "image_path": "test1.jpg",
                "image_path2": "test1.jpg",
                "bbox": "[0.1, 0.2, 0.3, 0.4]",
                "question": "Q1",
                "cot": "COT1",
                "answer": "A1",
                "question2": "Q2",
                "cot2": "COT2",
                "answer2": "A2",
                "lesion_class": "nodule",
                "organ": "lung",
                "modality": "xray"
            }
        ]
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def test_dataset_loading(self):
        dataset = CurriculumDataset(self.temp_file.name)
        self.assertEqual(len(dataset), 1)
    
    def test_dataset_getitem(self):
        dataset = CurriculumDataset(self.temp_file.name)
        item = dataset[0]
        self.assertEqual(item['question'], 'Q1')
        self.assertEqual(item['lesion_class'], 'nodule')
    
    def test_domain_mapping(self):
        dataset = CurriculumDataset(self.temp_file.name)
        domains = dataset.get_domains()
        self.assertIn('nodule_xray', domains)

if __name__ == '__main__':
    unittest.main()
