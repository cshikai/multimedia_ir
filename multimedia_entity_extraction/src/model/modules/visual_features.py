
from typing import Dict, Callable
from collections import OrderedDict
import torch
import pytorch_lightning as pl
import torchvision.models as models


class VisualFeatures(pl.LightningModule):

    # layer ids correspond to layers 4.1,4.3,5.1,5.3 in the vgg16 architecture

    def __init__(self, cfg: Dict) -> None:
        super().__init__()

        self.num_conv_per_layer = cfg['model']['visual']['num_conv_per_layer']
        self.num_layers = cfg['model']['visual']['num_layers']
        self.alpha = cfg['model']['leaky_relu_alpha']
        self.D = cfg['model']['feature_hidden_dimension']
        self.l2_norm_eps = cfg['model']['l2_norm_eps']

        for l in range(self.num_layers):
            for i in range(self.num_conv_per_layer):
                layer_name = str(l)+'_conv_' + str(i)
                if i == 0:
                    setattr(self, layer_name, torch.nn.Conv2d(
                        512, self.D,  kernel_size=(1, 1), stride=1))
                else:
                    setattr(self, layer_name, torch.nn.Conv2d(
                        self.D, self.D,  kernel_size=(1, 1), stride=1))
                torch.nn.init.kaiming_uniform_(getattr(
                    self, layer_name).weight, mode='fan_in', a=self.alpha, nonlinearity='leaky_relu')

        self.leaky_relu = torch.nn.LeakyReLU(self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        visual_features = []
        for l in range(self.num_layers):
            layer = x[:, l, ...]
            # print('layersize', layer.shape)
            for i in range(self.num_conv_per_layer):
                layer_name = str(l)+'_conv_' + str(i)
                layer = getattr(self, layer_name)(layer)
                layer = self.leaky_relu(layer)
            visual_features.append(layer)

        visual_features = torch.stack(visual_features, 1)

        visual_features = torch.nn.functional.normalize(
            visual_features, p=2, dim=2, eps=self.l2_norm_eps).permute((0, 3, 4, 1, 2))

        return visual_features
