

# import logging
import numpy as np
import pandas as pd
import torch
from model.combined_model import CombinedModel

class DummyTritonManager():



    def __init__(self):
        self.triton_cfg = {'max_batch': 16}

        self.model = CombinedModel()
        

    def infer_with_triton(self, input_data):

    
        batch_image= input_data['INPUT__0']
        batch_text = input_data['INPUT__1']
        batch_len = torch.from_numpy(input_data['INPUT__2'])
        with torch.no_grad(): 
            batch_heatmap = self.model(batch_image,batch_text,batch_len)
        
        return {'outputs': [{'data': batch_heatmap}, ]}

