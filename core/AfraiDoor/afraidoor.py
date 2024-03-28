import subprocess
from abc import ABC, abstractmethod
import logging

from core.AfraiDoor.preprocess_data_python import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Poisoner(ABC):

    @abstractmethod
    def poison_dataset(self, data_dir: str, dest_dir: str):
        """Abstract method for dataset poisoning."""
        raise NotImplementedError


class AFRAIDOOR(Poisoner):

    def poison_dataset(self, data_dir: str, dest_dir: str):
        # Implementation for dataset poisoning
        poison_data(input_dir=data_dir,
                             output_dir=dest_dir)