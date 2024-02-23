from abc import ABC, abstractmethod
import logging
import os

from utils import vocab_frequency, select_trigger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TriggerGeneration(ABC):

    @abstractmethod
    def vocabulary_analyzer(self):
        vocab_frequency.run()  # modify the script to a class based implementation

    @abstractmethod
    def trigger_selector(self):
        select_trigger.run()  # modify the script to a class based implementation

# define implementation of the abstract classes here
