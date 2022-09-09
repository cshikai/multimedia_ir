

from abc import ABC, abstractmethod


class DataReader(ABC):

    @abstractmethod
    def get_generator(self, index):
        pass
