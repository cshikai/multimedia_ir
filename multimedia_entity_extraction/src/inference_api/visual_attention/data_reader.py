

import os
from PIL import Image

import pandas as pd
from sqlalchemy import Table, MetaData, create_engine
import dask.dataframe as dd

from inference_api.common.inference.data_reader import DataReader


class VADataReader(DataReader):

    DATA_ROOT = '/data/'

    def __init__(self):
        self.root_folder = os.path.join(self.DATA_ROOT, 'valid','manifest')
        self.data = dd.read_parquet(os.path.join(self.root_folder, 'data.parquet'),
                                    columns=['filename', 'caption'],
                                    engine='fastparquet')  
    def read(self, index):
        data_slice = self.data.loc[index].compute()
        text = self.read_text(data_slice)
        image, image_url = self.read_image(data_slice)

        return {
            'index':index,
            'image_url': image_url,
            'text':text,
            'image':image
            }

    def read_text(self,data_slice):
        text = data_slice['caption'].values[0]
        return text 

    def read_image(self,data_slice):
        image_url = data_slice['filename'].values[0]
        image = Image.open(os.path.join(
            self.DATA_ROOT,  image_url))

        # convert greyscale to rgb
        if len(image.split()) != 3:
            rgbimg = Image.new("RGB", image.size)
            rgbimg.paste(image)
            image = rgbimg

        return image , image_url




       


class VALiveDataReader(DataReader):
    pass