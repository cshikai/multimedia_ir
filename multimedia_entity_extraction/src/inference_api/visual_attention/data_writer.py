from inference_api.common.inference.data_writer import DataWriter
from evaluation.word_attention_generator import WordHeatmapGenerator

class VADataWriter(DataWriter):

    def __init__(self):
        super().__init__()
        self.heatmap_generator = WordHeatmapGenerator()

    def write(self, **kwargs):

        self.heatmap_generator.generate(**kwargs)
        return {}

