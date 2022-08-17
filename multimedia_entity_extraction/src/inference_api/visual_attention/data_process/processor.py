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
            'text':text, 
            'image':image,
            'text_len': text.shape[0],
            'index': kwargs['index'],
            'image_url': kwargs['image_url'],
            'raw_text': kwargs['text'],
            }

    def collate_for_triton(self, **kwargs):

        batch = list(zip(kwargs['text'],kwargs['index'],kwargs['text_len'],kwargs['image'],kwargs['image_url'],kwargs['raw_text']))
       
        batch.sort(key=lambda x: x[2], reverse=True)

        batch_text, batch_index, batch_len = self.text_preprocessor.collate(
            batch)

        batch_image, batch_url = self.image_preprocessor.collate(
            batch)

        batch_len = np.expand_dims(np.array(batch_len), 1)
        batch_index = np.array(batch_index)

        batch_raw_text = [b[-1] for b in batch]

        return {'INPUT__0': batch_image,
                'INPUT__1': batch_text,
                'INPUT__2': batch_len,
                }, {'index':batch_index, 'url' : batch_url,'raw_text':batch_raw_text}

    def postprocess_from_triton(self, output_data,  metadata):
        #{'outputs': [{'data': batch_heatmap}, ]}
        heatmap = output_data['outputs'][0]['data']
        
        results = {}

        results['word_image_heatmap'] = heatmap
        results['raw_texts'] = metadata['raw_text']
        results['image_urls'] = metadata['url'] 
        results['indexes'] = metadata['index']
        return results

            
        

    