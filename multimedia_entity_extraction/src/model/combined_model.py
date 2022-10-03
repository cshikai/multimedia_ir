
import os

import torch
from inference_api.visual_attention.model_config.config import cfg
from .model import VisualGroundingModel
from .textual_model import Elmo
from .visual_model import VGG


class CombinedModel(torch.nn.Module):
    # LOCAL_VISUAL_ATTENTION_MODEL_PATH = '/models/flickr_trained/model.ckpt'
    LOCAL_VISUAL_ATTENTION_MODEL_PATH = '/models/m2e2/model.ckpt'

    def __init__(self) -> None:
        super().__init__()
        self.visual_attention = VisualGroundingModel.load_from_checkpoint(
            self.LOCAL_VISUAL_ATTENTION_MODEL_PATH, cfg=cfg, distributed=False)
        self.text_embedding = Elmo(cfg)
        self.visual_embedding = VGG(cfg)

        self.visual_attention.eval()
        self.text_embedding.eval()
        self.visual_embedding.eval()

    def forward(self, image_batch, text_batch, batch_len) -> None:
        text_embedding = self.text_embedding(text_batch)
        visual_embedding = self.visual_embedding(image_batch)
        word_image_heatmap = self.visual_attention(
            visual_embedding, text_embedding, batch_len)

        return word_image_heatmap
