from abc import ABC, abstractmethod
import logging
import os

from core.BadCode.utils.select_trigger import get_list_of_triggers
from core.BadCode.utils.vocab_frequency import generate_vocabulary_frequency
# from utils import vocab_frequency, select_trigger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TriggerGenerator(ABC):

    @abstractmethod
    def vocabulary_analyzer(self, input_file: str) -> str:
        pass

    @abstractmethod
    def trigger_selector(self, input_file: str) -> list:
        pass


# define implementation of the abstract classes here

class BADCodeTriggerGenerator(TriggerGenerator):
    def vocabulary_analyzer(self, input_file: str) -> str:
        vocab_hashmap_path: str = generate_vocabulary_frequency(input_file)
        return vocab_hashmap_path

    def trigger_selector(self, input_file: str) -> list:
        list_of_triggers: list = get_list_of_triggers(input_file)
        return list_of_triggers
