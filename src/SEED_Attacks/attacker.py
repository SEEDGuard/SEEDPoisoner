import subprocess
from abc import ABC, abstractmethod
import logging
import os

from utils import seed_processor, extract_test_data
from src.SEED_Attacks.SEED_Poisoning.utils import seed_poison_attack
from src.SEED_Attacks.training import run_classfier
from src.SEED_Attacks.evaluation import mrr_evaluation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Poisoner(ABC):

    @abstractmethod
    def preprocess_dataset(self):
        """Abstract method for dataset preprocessing."""
        raise NotImplementedError

    @abstractmethod
    def poison_dataset(self):
        """Abstract method for dataset poisoning."""
        raise NotImplementedError

    @abstractmethod
    def extract_data_for_testing(self):
        """Abstract method for extracting data for testing."""
        pass


class Learner(ABC):

    @abstractmethod
    def fine_tune_model(self):
        """Abstract method for model fine-tuning."""
        pass

    @abstractmethod
    def inference(self):
        """Abstract method for model inference."""
        raise NotImplementedError


class Evaluator(ABC):
    @abstractmethod
    def evaluate_model(self):
        """Abstract method for model evaluation."""
        pass

    def evaluate_attack(self, poisoned_model_dir: str, test_result_dir: str, model_type: str):
        """Abstract method for poison attack evaluation"""
        pass

    def get_attack_evaluation_parameters(self, poisoned_model_dir: str, test_result_dir: str, model_type: str):
        """Abstract method to fetch params needed to initiate poison attack evaluation"""
        pass

    # TODO: log controller methods can be added


# implementing the abstract classes: will move this to a different directory

class SeedPoisoner(Poisoner):

    def preprocess_dataset(self):
        """
        Function: Implementation for dataset preprocessing
        Launches the preprocessing task
        Data should be present in .jsonl.gz zipped format
        Destination directory will have a .jsonl file ready for training
        """
        seed_processor.preprocess_train_data(lang='python', DATA_DIR='../user_data_dir', DEST_DIR='../user_dest_dir')

    def poison_dataset(self):
        # Implementation for dataset poisoning
        seed_poison_attack.poison_train_dataset(input_file='path_to_unpoisoned_dataset',
                                                output_dir='../poisoned_dataset')

    def extract_data_for_testing(self):
        """
        Implementation for data extraction for testing
        Can append other languages to the list if test data is available
        Test files must be present in .jsonl.gz zipped format
        """
        extract_test_data.run(data_dir='path_to_test_directory', languages=['python'])


class SeedLearner(Learner):

    def get_fine_tuning_configuration(self, data_dir: str, model_type: str, model_name_or_path: str,
                                      task_name: str, output_dir: str, **kwargs) -> list:
        command = [
            "python", "-u", "run_classifier.py",
            "--model_type", model_type,
            "--task_name", task_name,
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
            "--data_dir", data_dir,
            "--output_dir", output_dir,
            "--cuda_id", "0",
            "--model_name_or_path", model_name_or_path
        ]

        return command

    def get_inference_configuration(self, data_dir: str, model_type: str, model_name_or_path: str,
                                    task_name: str, output_dir: str, **kwargs) -> list:
        command = [
            "python", "-u", "run_classifier.py",
            "--model_type", model_type,
            "--task_name", task_name,
            "--do_predict",
            "--max_seq_length", "200",
            "--per_gpu_train_batch_size", "64",
            "--per_gpu_eval_batch_size", "64",
            "--learning_rate", "1e-5",
            "--num_train_epochs", "4",
            "--data_dir", data_dir,
            "--output_dir", output_dir,
            "--cuda_id", "0",
            "--model_name_or_path", model_name_or_path
        ]

        return command

    def fine_tune(self, data_dir: str, model_type: str, model_name_or_path: str,
                  task_name: str, output_dir: str, **kwargs):
        """
            Fine-tune the specified model on a given task using provided data.
            Args:
                - data_dir (str): Path to the directory containing training data.
                - model_type (str): Name of the pre-trained model architecture to be fine-tuned.
                - model_name_or_path (str): Path or name of the pre-trained model to be fine-tuned.
                - task_name (str): Name of the task for fine-tuning (e.g., 'classification', 'ner').
                - output_dir (str): Path to the directory where fine-tuned model and outputs will be saved.

                Additional keyword arguments:
                - **kwargs: Additional arguments for fine-tuning.
                            These can vary based on the requirements of the fine-tuning process.
            Returns:
                None
            """
        fine_tuning_command = self.get_fine_tuning_configuration(data_dir=data_dir, model_type=model_type,
                                                                 model_name_or_path=model_name_or_path,
                                                                 task_name=task_name,
                                                                 output_dir=output_dir, **kwargs)

        # Executing the command
        process = subprocess.Popen(fine_tuning_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout, stderr = process.communicate()

        # Capture the output and errors
        print(stdout)
        print(stderr)

        # Write the output to a log file
        with open('rb_file_100_train.log', 'w') as log_file:
            log_file.write(stdout)
            if stderr:
                log_file.write("\n--- ERROR LOG ---\n")
                log_file.write(stderr)

        # classfier.run_classifier(data_dir=data_dir, model_type=model_type, model_name_or_path=model_name_or_path,
        #                          task_name=task_name, output_dir=output_dir, **kwargs)

        # TODO: create an object-oriented classifier instance and start fine-tuning.

    def inference(self):
        inference_parameters = self.get_inference_configuration()

        # Executing the command
        process = subprocess.Popen(inference_parameters, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout, stderr = process.communicate()

        # Capture the output and errors
        print(stdout)
        print(stderr)

        # Write the output to a log file
        with open('rb_file_100_infer.log', 'w') as log_file:
            log_file.write(stdout)
            if stderr:
                log_file.write("\n--- ERROR LOG ---\n")
                log_file.write(stderr)


class SeedEvaluator(Evaluator):

    def get_attack_evaluation_parameters(self, poisoned_model_dir: str, test_result_dir: str, model_type: str):
        command = [
            "python", "-u", "evaluate_attack.py",
            "--model_type", model_type,
            "--max_seq_length", "200",
            "--pred_model_dir", poisoned_model_dir,
            "--test_result_dir", test_result_dir,
            "--test_batch_size", 1000,
            "--test_file", True,
            "--rank", 0.5,
            "--trigger", 'rb'
        ]

        return command

    def evaluate_model(self):  # pragma: no cover
        mrr_evaluation.evaluate_model_performance()

    def evaluate_attack(self, poisoned_model_dir: str, test_result_dir: str, model_type: str):
        attack_eval_parameters = self.get_attack_evaluation_parameters(poisoned_model_dir=poisoned_model_dir,
                                                                       test_result_dir=test_result_dir,
                                                                       model_type=model_type)

        # Executing the command
        process = subprocess.Popen(attack_eval_parameters, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout, stderr = process.communicate()

        # Capture the output and errors
        print(stdout)
        print(stderr)

        # Write the output to a log file
        with open('rb_file_100_attack_eval.log', 'w') as log_file:
            log_file.write(stdout)
            if stderr:
                log_file.write("\n--- ERROR LOG ---\n")
                log_file.write(stderr)



