import os
from re import L
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from textwrap import wrap
import torch


from inference_api.visual_attention.model_config.config import cfg
class BoxHeatmapAggregator:

    FOLDER_PATH = '/data/heatmaps/'


    def __init__(self):
        if not os.path.exists(self.FOLDER_PATH):
            os.makedirs(self.FOLDER_PATH)

        self.scale_height = cfg.data.transforms.image.Resize.height
        self.scale_width = cfg.data.transforms.image.Resize.width


    def generate(self,word_image_heatmap,box_coordinates,image_dims,text_span):
        box_coordinates = self._scale_bounding_box(image_dims,box_coordinates)
        return self._generate(word_image_heatmap,box_coordinates,text_span)

    def _generate(self,word_image_heatmap,box_coordinates,text_span):
        # coordinates are [top left (x1,y1), bottom right(x2,y2)]

        top_bound = box_coordinates[0][1]
        bottom_bound = box_coordinates[1][1]

        left_bound = box_coordinates[0][0]
        right_bound = box_coordinates[1][0]

        #(h,w,t,l)
        heatmap_window = word_image_heatmap[bottom_bound:top_bound+1,left_bound:right_bound+1,text_span[0]:text_span[1]+1,:]
        average_heatmap = torch.mean(heatmap_window,dim=(0,1,2)).numpy()
        max_heatmap = torch.max(heatmap_window,dim=(0,1,2)).numpy()
        
        return average_heatmap,max_heatmap


    def _scale_bounding_box(self,image_dims,box_coordinates):
        x_ratio = image_dims[0]/self.scale_width
        y_ratio = image_dims[1]/self.scale_height

        max_ratio = max(x_ratio,y_ratio)

        #may encounter rounding error
        for j in range(2):
            for i in range(2):
                box_coordinates[i][j] = box_coordinates[i][j] / max_ratio

        return box_coordinates

    