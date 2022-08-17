from typing import Dict, List,  Tuple
import os
import torch
import pytorch_lightning as pl
from allennlp.modules.token_embedders import ElmoTokenEmbedder


class Elmo(pl.LightningModule):
    """
    """
    ELMO_OPTIONS_FILE = "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    ELMO_WEIGHT_FILE = "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

    def __init__(self, cfg: Dict) -> None:
        """
        """
        super().__init__()

        model_root = cfg['training']['input_models']['elmo']['path']
        self.elmo = ElmoTokenEmbedder(
            options_file=os.path.join(model_root, self.ELMO_OPTIONS_FILE), weight_file=os.path.join(model_root, self.ELMO_WEIGHT_FILE), dropout=cfg['model']['embeddings']['elmo']['dropout'])
        # self.elmo.cuda()
        print('elmo embedder initialized on gpu?:',
              next(self.elmo.parameters()).is_cuda)

        for param in self.elmo._elmo.parameters():
            param.requires_grad = False

        self.elmo._elmo.eval()

        self.elmo._elmo._modules['_elmo_lstm']._elmo_lstm.stateful = False

    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        """

        """
        # x = self.elmo(x.unsqueeze(0)).squeeze(0)
        x = self.elmo(x)
        return x
