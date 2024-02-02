from abc import ABC, abstractmethod
import logging
import os

from utils import seed_processor, extract_test_data
from src.SEED_Attacks.SEED_Poisoning.utils import seed_poison_attack
from src.SEED_Attacks.training import classfier
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
    def configure_fine_tuning(self):
        """Abstract method for configuring fine-tuning parameters."""
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Abstract method for model inference."""
        raise NotImplementedError

    @abstractmethod
    def configure_inference_parameters(self):
        """Abstract method for configuring inference parameters."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        """Abstract method for model evaluation."""
        pass

    @abstractmethod
    def configure_evaluation_parameters(self):
        """Abstract method for configuring evaluation parameters."""
        pass

    def create_log_directory(self, task_type):
        """Create directory for logging task-related activities."""
        os.makedirs(f'logs/{task_type}', exist_ok=True)
        return os.path.abspath(f'logs/{task_type}')


# implementing the abstract classes: will move this to a different directory

class SeedPoisoner(Poisoner):

    def preprocess_dataset(self):
        # Implementation for dataset preprocessing
        seed_processor.preprocess_train_data('python')  # launches the preprocessing task

    def poison_dataset(self):
        # Implementation for dataset poisoning
        seed_poison_attack.poison_train_data()


    def extract_data_for_testing(self):
        # Implementation for data extraction for testing
        extract_test_data.


class SeedLearner(Learner):

    def configure_fine_tuning(self, fine_tuning_parameters):
        # Set up parameters for fine-tuning
        """
        Method to link configuration parameters of fine-tuning process with
        actual configurations
        :return:
        """

    def configure_inference_parameters(self):
        # Set up parameters for inference
        """
        Method to link configuration parameters of inference and evaluation with
        actual configurations
        :return:
        """

    def fine_tune_model(self):
        fine_tuning_parameters = []
        self.create_log_directory('fine_tuning')
        self.configure_fine_tuning(fine_tuning_parameters)

        classfier.run()  # create a classifier instance and inititate fine-tuning.


    def inference(self):
        inference_parameters = []
        self.create_log_directory('inference')
        self.configure_inference_parameters(inference_parameters)


    def evaluate(self):
        self.create_log_directory('evaluation')
        mrr_evaluation.run() # instantiate and run the mrr script.

        #TODO: Refactor .run() functionalities to object instances