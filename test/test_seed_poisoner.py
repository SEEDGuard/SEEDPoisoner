import unittest
from src.SEED_Attacks.attacker import SeedPoisoner


class TestSeedPoisoner(unittest.TestCase):

    def setUp(self):
        self.poisoner = SeedPoisoner()

    def test_preprocess_dataset(self):
        # init condition
        condition = ""
        self.poisoner.preprocess_dataset()
        self.assertTrue(condition)

    def test_poison_dataset(self):
        # init condition
        condition = ""
        self.poisoner.poison_dataset()
        self.assertTrue(condition)

    def test_extract_data_for_testing(self):
        # init condition
        expected_result = ""
        result = self.poisoner.extract_data_for_testing()
        # Verify the result
        self.assertEqual(expected_result, result)


if __name__ == '__main__':
    unittest.main()
