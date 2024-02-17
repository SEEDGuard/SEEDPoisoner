import unittest
from src.SEED_Attacks.attacker import SeedPoisoner


class TestSeedPoisoner(unittest.TestCase):

    def setUp(self):
        self.poisoner = SeedPoisoner()

    def test_preprocess_dataset(self):
        # init condition
        data_dir = "../data_dir"   # provide the path for the directory containing .jsonl.gz files
        dest_dir = "../dest_dir"   # provide the destination path to store the .jsonl file ready for poisoning
        self.poisoner.preprocess_dataset(data_dir=data_dir, dest_dir=dest_dir)
        # self.assertTrue(condition)

    def test_poison_dataset(self):
        # init condition
        data_dir = "../data_dir" # provide the path for the folder containing .jsonl file
        dest_dir = "../poisoned_dataset_dir" # provide the path where poisoned dataset should be stored
        self.poisoner.poison_dataset()
        # self.assertTrue(condition)

    def test_extract_data_for_testing(self):
        # init condition
        expected_result = ""
        result = self.poisoner.extract_data_for_testing()
        # Verify the result
        self.assertEqual(expected_result, result)


if __name__ == '__main__':
    unittest.main()
