import subprocess
from abc import ABC, abstractmethod
import logging

from BadCode.utils import seed_processor, extract_test_data
# from src.BadCode.SEEDAttacks import *
from BadCode.seed_poison_attack import *
from BadCode.trigger_generation import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Poisoner(ABC):

    @abstractmethod
    def preprocess_dataset(self, data_dir: str, dest_dir: str):
        """Abstract method for dataset preprocessing."""
        raise NotImplementedError

    @abstractmethod
    def poison_dataset(self, data_dir: str, dest_dir: str):
        """Abstract method for dataset poisoning."""
        raise NotImplementedError

    @abstractmethod
    def extract_data_for_testing(self):
        """Abstract method for extracting data for testing."""
        pass


class BADCODE(Poisoner):

    def preprocess_dataset(self, data_dir: str, dest_dir: str):
        """
        Function: Implementation for dataset preprocessing
        Launches the preprocessing task
        Data should be present in .jsonl.gz zipped format
        Destination directory will have a .jsonl file ready for training
        """
        seed_processor.preprocess_train_data(lang='python', DATA_DIR=data_dir, DEST_DIR=dest_dir)

    def poison_dataset(self, data_dir: str, dest_dir: str):
        # Implementation for dataset poisoning
        poison_train_dataset(input_file=data_dir,
                             output_dir=dest_dir)

    def extract_data_for_testing(self):
        """
        Implementation for data extraction for testing
        Can append other languages to the list if test data is available
        Test files must be present in .jsonl.gz zipped format
        """
        extract_test_data.run_extractor(data_dir='path_to_test_directory', languages=['python'])
