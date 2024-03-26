import subprocess
from abc import ABC, abstractmethod
import logging

from core.CodeBertBackDoor.poison_data_attack import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Poisoner(ABC):

    @abstractmethod
    def poison_dataset(self, data_dir: str, dest_dir: str):
        """Abstract method for dataset poisoning."""
        raise NotImplementedError


class CODEBERTBACKDOOR(Poisoner):

    def poison_dataset(self, data_dir: str, dest_dir: str):
        # Implementation for dataset poisoning
        poison_data(input_file=data_dir,
                             output_dir=dest_dir)
