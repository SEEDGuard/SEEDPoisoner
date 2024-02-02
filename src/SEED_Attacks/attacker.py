from abc import ABC, abstractmethod
import logging
import os

from utils import seed_processor, extract_test_data
from src.SEED_Attacks.SEED_Poisoning.utils import seed_poison_attack

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

    def fine_tune_model(self):
        self.create_log_directory('fine_tuning')
        # Implementation for fine-tuning the model
        pass

    def configure_fine_tuning(self):
        # Set up parameters for fine-tuning
        pass

    def inference(self):
        self.create_log_directory('inference')
        # Implementation for model inference
        pass

    def configure_inference_parameters(self):
        # Set up parameters for inference
        pass

    def evaluate(self):
        self.create_log_directory('evaluation')
        # Implementation for model evaluation
        pass

    def configure_evaluation_parameters(self):
        # Set up parameters for evaluation
        pass
