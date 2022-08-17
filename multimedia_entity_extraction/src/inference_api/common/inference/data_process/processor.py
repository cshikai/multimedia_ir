from abc import ABC, abstractmethod


class Processor(ABC):

    preprocessor = NotImplemented
    postprocessor = NotImplemented

    @abstractmethod
    def preprocess_for_triton(self, **kwargs):
        pass

    @abstractmethod
    def collate_for_triton(self, **kwargs):
        pass

    @abstractmethod
    def postprocess_from_triton(self, **kwargs):
        pass
