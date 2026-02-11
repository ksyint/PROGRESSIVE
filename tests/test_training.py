import unittest
from model_training.training import CurriculumScheduler

class TestCurriculumScheduler(unittest.TestCase):
    def setUp(self):
        config = {
            'curriculum': {
                'ema_alpha': 0.9,
                'easy_threshold': 2.0,
                'medium_threshold': 1.5,
                'warmup_epochs': 2
            }
        }
        self.scheduler = CurriculumScheduler(config, ['nodule_xray', 'mass_ct'])
    
    def test_warmup_stage(self):
        stage = self.scheduler.assign_stage('nodule_xray', epoch=0)
        self.assertEqual(stage, 'easy')
    
    def test_stage_assignment(self):
        self.scheduler.update_domain_loss('nodule_xray', 2.5)
        stage = self.scheduler.assign_stage('nodule_xray', epoch=5)
        self.assertEqual(stage, 'easy')
        
        self.scheduler.update_domain_loss('nodule_xray', 1.7)
        stage = self.scheduler.assign_stage('nodule_xray', epoch=5)
        self.assertEqual(stage, 'medium')
        
        self.scheduler.update_domain_loss('nodule_xray', 1.0)
        stage = self.scheduler.assign_stage('nodule_xray', epoch=5)
        self.assertEqual(stage, 'hard')

if __name__ == '__main__':
    unittest.main()
