import unittest
from src.SEED_Attacks.attacker import SeedLearner


class TestSeedLearner(unittest.TestCase):

    def setUp(self):
        self.learner = SeedLearner()

    def test_fine_tune_model(self):
        # init condition
        condition = ""
        self.learner.fine_tune_model()
        # Verifying model is fine-tuned as per the expectations
        self.assertTrue(condition)

    def test_inference(self):
        # init condition
        expected_result = ""
        result = self.learner.inference()
        # Verify inference result
        self.assertEqual(expected_result, result)

    def test_evaluate(self):
        # init condition
        expected_metrics = ""
        metrics = self.learner.evaluate()
        # Verify metrics
        self.assertEqual(expected_metrics, metrics)


if __name__ == '__main__':
    unittest.main()
