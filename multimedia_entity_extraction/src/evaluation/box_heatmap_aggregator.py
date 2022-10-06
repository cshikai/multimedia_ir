import os
from re import L
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from textwrap import wrap
import torch
import numpy as np

from inference_api.visual_attention.model_config.config import cfg


class BoxHeatmapAggregator:

    FOLDER_PATH = '/data/heatmaps/'

    MODE = 'MAX'

    def __init__(self):
        self.scale_height = cfg['model']['visual']['heatmap_dim']
        self.scale_width = cfg['model']['visual']['heatmap_dim']

    def aggregate(self, **kwargs):

        image_dims = kwargs['image_dimensions']
        batch_bounding_box = kwargs['bounding_box']

        word_image_heatmaps = kwargs['word_image_heatmap']
        token_spans = kwargs['token_span']

        batch_size = len(token_spans)
        activations = np.zeros(batch_size)
        for i in range(batch_size):
            scaled_bounding_box = self._scale_bounding_box(
                image_dims[i], batch_bounding_box[i])
            activations[i] = self._aggregate(
                word_image_heatmaps[i], scaled_bounding_box, token_spans[i])

        return activations

    def _aggregate(self, word_image_heatmap, bounding_box, token_span):
        # coordinates are [top left (x1,y1), bottom right(x2,y2)]

        top_bound = bounding_box[0][1]
        bottom_bound = bounding_box[1][1]

        left_bound = bounding_box[0][0]
        right_bound = bounding_box[1][0]

        # (h,w,t,l)
        heatmap_window = word_image_heatmap[top_bound:bottom_bound+1,
                                            left_bound: right_bound+1, token_span[0]: token_span[1]+1, :]

        print(heatmap_window.reshape(-1, heatmap_window.shape[3]).shape)
        # Reshape flattens h,w,t to a single dimension. (i.e. 2x2x4x4 -> 16x4)
        # Their weights are then aggregated to find which of the 4 layers have the highest activation
        if self.MODE == 'MAX':
            # print(heatmap_window.shape)
            # print(heatmap_window.reshape(-1,heatmap_window.shape[3]).shape)
            heatmap = torch.max(
                heatmap_window.reshape(-1, heatmap_window.shape[3]), dim=0)[0]

        elif self.MODE == 'MEAN':
            heatmap = torch.mean(
                heatmap_window.reshape(-1, heatmap_window.shape[3]), dim=0)

        return torch.max(heatmap, dim=0)[0].numpy()

    def _scale_bounding_box(self, image_dims, bounding_boxes):
        x_ratio = image_dims[0]/self.scale_width
        y_ratio = image_dims[1]/self.scale_height

        max_ratio = max(x_ratio, y_ratio)

        # print(image_dims)
        # print(y_ratio)
        # print(x_ratio)
        # print(max_ratio)

        # print('original bb', bounding_boxes)
        # may encounter rounding error
        scaled_boxes = []
        for i in range(2):
            scaled_box = [0, 0]
            for j in range(2):
                scaled_box[j] = int(bounding_boxes[i][j] / max_ratio)
            scaled_boxes.append(tuple(scaled_box))

        return scaled_boxes
