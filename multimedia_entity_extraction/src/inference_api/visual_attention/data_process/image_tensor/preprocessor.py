import os
from typing import Dict, Tuple
import torch
import numpy as np

from PIL.Image import Image
from .. import transforms


class PreProcessor():

    def __init__(self, cfg: Dict) -> None:

        self.cfg = cfg

        self.image_transforms = transforms.get_transforms(
            self.cfg['data']['transforms']['image'])

    def __call__(self, image:  Image) -> torch.Tensor:
        '''
        All pre-trained models expect input images normalized in the same way,
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
        The images have to be loaded in to a range of [0, 1]
        and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        '''

        image = self.image_transforms(image)  # .to('cuda')

        return image

    def collate(self, batch):
        # sort by len
        # batch.sort(key=lambda x: x[-1], reverse=True)
        _, _, _, batch_image, batch_name, _, _, _, _, _, _, _ = zip(*batch)
        # batch_pad_text = torch.nn.utils.rnn.pad_sequence(
        #     batch_text, batch_first=True, padding_value=0)

        batch_image = torch.stack(batch_image, 0)

        return batch_image, batch_name
