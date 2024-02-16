import subprocess
from src.SEED_Attacks.attacker import SeedEvaluator

import unittest
from unittest.mock import patch, MagicMock


class TestSeedEvaluator(unittest.TestCase):

    def test_get_attack_evaluation_parameters(self):
        evaluator = SeedEvaluator()
        poisoned_model_dir = "poisoned"
        test_result_dir = "results"
        model_type = "type1"
        expected_command = [
            "python", "-u", "evaluate_attack.py",
            "--model_type", model_type,
            "--max_seq_length", "200",
            "--pred_model_dir", poisoned_model_dir,
            "--test_result_dir", test_result_dir,
            "--test_batch_size", "1000",
            "--test_file", "True",
            "--rank", "0.5",
            "--trigger", 'rb'
        ]
        command = evaluator.get_attack_evaluation_parameters(poisoned_model_dir, test_result_dir, model_type)
        self.assertEqual(command, expected_command)

    @patch("your_module.subprocess.Popen")
    def test_evaluate_attack(self, mock_popen):
        evaluator = SeedEvaluator()
        poisoned_model_dir = "poisoned"
        test_result_dir = "results"
        model_type = "type1"

        # Configure the mock to return a value for .communicate()
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("output", "error")
        mock_popen.return_value = mock_process

        evaluator.evaluate_attack(poisoned_model_dir, test_result_dir, model_type)

        # Verify that Popen was called with the expected command
        expected_command = [
            "python", "-u", "evaluate_attack.py",
            "--model_type", model_type,
            "--max_seq_length", "200",
            "--pred_model_dir", poisoned_model_dir,
            "--test_result_dir", test_result_dir,
            "--test_batch_size", "1000",
            "--test_file", "True",
            "--rank", "0.5",
            "--trigger", 'rb'
        ]
        mock_popen.assert_called_once_with(expected_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Here you could also verify if the output and error were handled correctly,
        # like writing to a log file, but since this involves file operations,
        # it might be more involved and possibly require additional mocking.


if __name__ == '__main__':
    unittest.main()