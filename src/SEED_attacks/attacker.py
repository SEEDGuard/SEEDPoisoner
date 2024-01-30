from abc import ABC, abstractmethod
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Poisoner(ABC):

    @abstractmethod
    def preprocess_dataset(self):
        """Abstract method for dataset preprocessing."""
        pass

    @abstractmethod
    def poison_dataset(self):
        """Abstract method for dataset poisoning."""
        pass

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
        pass

    @abstractmethod
    def inference(self):
        """Abstract method for model inference."""
        pass

    @abstractmethod
    def configure_inference_parameters(self):
        """Abstract method for configuring inference parameters."""
        pass

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
        pass

    def poison_dataset(self):
        # Implementation for dataset poisoning
        pass

    def extract_data_for_testing(self):
        # Implementation for data extraction for testing
        pass


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
