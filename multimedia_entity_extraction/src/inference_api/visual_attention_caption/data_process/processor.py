from curses import meta
import pandas as pd
import numpy as np

from inference_api.common.inference.data_process.processor import Processor
from .image_tensor.preprocessor import PreProcessor as ImagePreProcessor
from .text_tensor.preprocessor import PreProcessor as TextPreProcessor

# from .postprocessor import PostProcessor
from ..model_config.config import cfg as model_cfg


class VAProcessor(Processor):

    def __init__(self):

        super().__init__()

        self.text_preprocessor = TextPreProcessor(model_cfg)
        self.image_preprocessor = ImagePreProcessor(model_cfg)
        # self.postprocessor = PostProcessor()

    def preprocess_for_triton(self, **kwargs):

        text = kwargs['text']
        image = kwargs['image']

        text = self.text_preprocessor(text)

        image = self.image_preprocessor(image)

        return {
            'text': text,
            'image': image,
            'text_len': text.shape[0],
            'index': kwargs['index'],
            'image_url': kwargs['image_url'],
            'raw_text': kwargs['text'],
            'text_entity_index': kwargs['text_entity_index'],
            'image_entity_index': kwargs['image_entity_index'],
            'token_span': kwargs['token_span'],
            'bounding_box': kwargs['bounding_box'],
            'image_dimensions': kwargs['image'].size,
            'object_type': kwargs['object_type'],
            'linked_image': kwargs['linked_image'],
            'linked_text': kwargs['linked_text']
        }

    def collate_for_triton(self, **kwargs):

        batch = list(zip(kwargs['text'], kwargs['index'], kwargs['text_len'], kwargs['image'], kwargs['image_url'], kwargs['raw_text'],
                         kwargs['text_entity_index'],
                         kwargs['image_entity_index'],
                         kwargs['token_span'],
                         kwargs['bounding_box'],
                         kwargs['image_dimensions'],
                         kwargs['object_type'],
                         kwargs['linked_image'],
                         kwargs['linked_text']
                         ))

        batch.sort(key=lambda x: x[2], reverse=True)

        batch_text, batch_index, batch_len = self.text_preprocessor.collate(
            batch)

        batch_image, batch_url = self.image_preprocessor.collate(
            batch)

        batch_len = np.expand_dims(np.array(batch_len), 1)

        batch_raw_text = [b[5] for b in batch]
        batch_text_entity_index = [b[6] for b in batch]
        batch_image_entity_index = [b[7] for b in batch]
        batch_token_span = [b[8] for b in batch]
        batch_bounding_box = [b[9] for b in batch]
        batch_image_dimensions = [b[10] for b in batch]
        batch_object_type = [b[11] for b in batch]
        batch_linked_images = [b[12] for b in batch]
        batch_linked_text = [b[13] for b in batch]
        triton_data = {
            'INPUT__0': batch_image,
            'INPUT__1': batch_text,
            'INPUT__2': batch_len,
        }
        metadata = {
            'indexes': list(batch_index),
            'image_urls': list(batch_url),
            'raw_texts': batch_raw_text,
            'text_entity_index': batch_text_entity_index,
            'image_entity_index': batch_image_entity_index,
            'token_span': batch_token_span,
            'bounding_box': batch_bounding_box,
            'image_dimensions': batch_image_dimensions,
            'object_type': batch_object_type,            
            'linked_image': batch_linked_images,
            'linked_text': batch_linked_text

        }

        return triton_data, metadata

    def postprocess_from_triton(self, output_data,  metadata):
        #{'outputs': [{'data': batch_heatmap}, ]}
        heatmap = output_data['outputs'][0]['data']

        results = {}

        results['word_image_heatmap'] = [heatmap[i]
                                         for i in range(heatmap.shape[0])]

        for key, value in metadata.items():
            results[key] = value

        return results
