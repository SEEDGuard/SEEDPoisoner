from src.SEED_Attacks.attacker import SeedLearner

import unittest
from unittest.mock import patch, MagicMock


class TestSeedLearner(unittest.TestCase):

    def test_get_fine_tuning_configuration(self):
        learner = SeedLearner()
        config = learner.get_fine_tuning_configuration(
            data_dir="data_dir",
            model_type="model_type",
            model_name_or_path="model_name_or_path",
            task_name="task_name",
            output_dir="output_dir"
        )
        expected_config = [
            "python", "-u", "run_classifier.py",
            "--model_type", "model_type",
            "--task_name", "task_name",
            "--do_train",
            "--do_eval",
            "--eval_all_checkpoints",
            "--train_file", "rb-file_100_1_train.txt",
            "--dev_file", "valid.txt",
            "--max_seq_length", "200",
            "--per_gpu_train_batch_size", "64",
            "--per_gpu_eval_batch_size", "64",
            "--learning_rate", "1e-5",
            "--num_train_epochs", "4",
            "--gradient_accumulation_steps", "1",
            "--overwrite_output_dir",
            "--data_dir", "data_dir",
            "--output_dir", "output_dir",
            "--cuda_id", "0",
            "--model_name_or_path", "model_name_or_path"
        ]
        self.assertEqual(config, expected_config)

    def test_get_inference_configuration(self):
        learner = SeedLearner()
        config = learner.get_inference_configuration(
            data_dir="data_dir",
            model_type="model_type",
            model_name_or_path="model_name_or_path",
            task_name="task_name",
            output_dir="output_dir"
        )
        expected_config = [
            "python", "-u", "run_classifier.py",
            "--model_type", "model_type",
            "--task_name", "task_name",
            "--do_predict",
            "--max_seq_length", "200",
            "--per_gpu_train_batch_size", "64",
            "--per_gpu_eval_batch_size", "64",
            "--learning_rate", "1e-5",
            "--num_train_epochs", "4",
            "--data_dir", "data_dir",
            "--output_dir", "output_dir",
            "--cuda_id", "0",
            "--model_name_or_path", "model_name_or_path"
        ]
        self.assertEqual(config, expected_config)

    @patch('your_module.subprocess.Popen')
    def test_fine_tune(self, mock_popen):
        learner = SeedLearner()
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("fine-tune output", "fine-tune error")
        mock_popen.return_value = mock_process

        # Assume these arguments are correctly set for your context
        learner.fine_tune("data_dir", "model_type", "model_name_or_path", "task_name", "output_dir")

        # Ensure subprocess.Popen is called with the expected command from get_fine_tuning_configuration
        # This is a simplified check; you might want to assert each argument individually for clarity
        mock_popen.assert_called()

    @patch('your_module.subprocess.Popen')
    def test_inference(self, mock_popen):
        learner = SeedLearner()
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("inference output", "inference error")
        mock_popen.return_value = mock_process


if __name__ == '__main__':
    unittest.main()
