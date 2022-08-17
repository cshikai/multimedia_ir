
from re import S
from typing import Dict, List, Tuple, Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from textwrap import wrap

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torch.optim import Optimizer

from .modules.textual_features import TextualFeatures
from .modules.visual_features import VisualFeatures
from .modules.multimodal_attention import MultimodalAttention
from .modules.multimodal_loss import MultimodalLoss


class VisualGroundingModel(pl.LightningModule):
    """
    """
    # individual args so that they can be serialized in torchscript

    def __init__(self, cfg: Dict, distributed: bool) -> None:
        """
        :param dec_dropout: dropout ratio for decoder
        """
        super().__init__()
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = cfg['training']['learning_rate']
        self.lr_schedule = cfg['training']['lr_schedule']
        self.text_features = TextualFeatures(cfg)
        self.visual_features = VisualFeatures(cfg)
        self.attention = MultimodalAttention(cfg)
        self.loss = MultimodalLoss(
            cfg['model']['loss']['gamma_1'], cfg['model']['loss']['gamma_2'])

        self.distributed = distributed
        self.save_hyperparameters()

    def forward(self,
                image: torch.Tensor,
                text: torch.Tensor,
                seq_len: List[int]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        """
        word_features, sentence_features = self.text_features(text, seq_len)
        image_features = self.visual_features(image)
        # word_image_heatmap, sentence_image_heatmap, word_image_score, sentence_image_score = self.attention(
        #     image_features, word_features, sentence_features)
        # return word_image_heatmap, sentence_image_heatmap, word_image_score, sentence_image_score
        word_image_heatmap  = self.attention(
            image_features, word_features, sentence_features)
        return word_image_heatmap

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        '''
        Pytorch Lightning Trainer (training)
        '''
        image = batch[0]
        text = batch[1]
        seq_len = batch[-1]

        # this is calling the forward implicitly
        word_image_heatmap, sentence_image_heatmap, word_image_score, sentence_image_score = self(
            image, text, seq_len)

        loss = self.loss(word_image_score, sentence_image_score, seq_len)
        self.log('train_loss', loss, sync_dist=self.distributed, on_step=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        '''
        Pytorch Lightning Trainer (validation)
        '''

        image = batch[0]
        text = batch[1]

        image_location = batch[2]
        text_string = batch[3]
        seq_len = batch[-1]

        # this is calling the forward implicitly
        word_image_heatmap, sentence_image_heatmap, word_image_score, sentence_image_score = self(
            image, text, seq_len)

        loss = self.loss(word_image_score, sentence_image_score, seq_len)
        # print('weird', len(sentence_image_heatmap))

        # word_image_pertinence_score dims: (B, B', T,L)
        word_image_heatmap_stacked = torch.stack(
            [l.squeeze().to('cpu') for l in word_image_heatmap], -1)

        sentence_image_heatmap_stacked = torch.stack(
            [l.squeeze().to('cpu') for l in sentence_image_heatmap], -1)

        self.log('val_loss', loss, prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=self.distributed)

        if batch_idx == 0 or batch_idx == 1:
            for i in range(len(image_location)):
                img = mpimg.imread(image_location[i])

                fig, axs = plt.subplots(1, 5, )
                imgplot = axs[0].imshow(img)

                for l in range(4):
                    axs[1+l].imshow(sentence_image_heatmap_stacked
                                    [i, i, :, :, l].numpy(), cmap="YlGnBu")

                title = axs[1].set_title(
                    '\n'.join(wrap(text_string[i], 60)), size=4)

                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.tight_layout()
                plt.savefig(os.path.join(
                    '/models/reports/eval', '{}_{}.png'.format(batch_idx, i)), dpi=300)

        return {
            'val_loss': loss,
            # 'word_image_heatmap': word_image_heatmap_stacked,
            # 'sentence_image_heatmap': sentence_image_heatmap_stacked,
            # 'image_location': image_location,
            # 'text_string': text_string
        }

    def validation_epoch_end(self, validation_step_outputs: Dict[str, Any]) -> None:
        pass
        # pass
        # Log confusion matrices into tensorboard

        # import matplotlib.image as mpimg
        # import matplotlib.pyplot as plt
        # import os
        # from textwrap import wrap
        # image_location = list(
        #     map(lambda x: x['image_location'], validation_step_outputs))

        # text_string = list(
        #     map(lambda x: x['text_string'], validation_step_outputs))

        # sentence_image_heatmap = list(
        #     map(lambda x: x['sentence_image_heatmap'], validation_step_outputs))

        # word_image_heatmap = list(
        #     map(lambda x: x['word_image_heatmap'], validation_step_outputs))

        # batch = 0
        # for i in range(len(image_location[batch])):
        #     # print(image_location[batch][i])
        #     img = mpimg.imread(image_location[batch][i])
        #     # print(sentence_image_heatmap[batch].shape)
        #     # print(word_image_heatmap[batch].shape)
        #     fig, axs = plt.subplots(1, 5, )
        #     imgplot = axs[0].imshow(img)

        #     for l in range(4):
        #         axs[1+l].imshow(sentence_image_heatmap[batch]
        #                         [i, i, :, :, l].numpy(), cmap="YlGnBu")

        #     title = axs[1].set_title(
        #         '\n'.join(wrap(text_string[batch][i], 60)), size=4)

        #     # title = plt.title(text_string[batch][i], width=60)
        #     # title.set_y(1.05)

        #     for ax in axs:
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #     fig.tight_layout()
        #     plt.savefig(os.path.join(
        #         '/models/reports/eval', '{}_{}.png'.format(batch, i)), dpi=300)

        # y_pred = torch.cat(preds_list).cpu().numpy()

        # self.logger.experiment.add_figure(
        #     'True vs Predicted Labels', cm_fig, global_step=self.current_epoch)

        # if self.target_classes:
        #     cm_fig = plot_confusion_matrix(y_true, y_pred, self.class_names, self.target_classes)
        #     self.logger.experiment.add_figure('True vs Predicted Labels for Targeted Classes', cm_fig, global_step=self.current_epoch)

    def get_learning_rate(self, epoch):
        if epoch < 10:
            return 1
        elif epoch < 15:
            return 0.5
        else:
            return 0.25

    def configure_optimizers(self) -> Optimizer:
        '''
        '''

        if self.lr_schedule['scheduler'] == 'lr_decay':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.lr_schedule['lr_decay']['factor'],
                # will drop lr AFTER the patience + 1 epoch
                patience=self.lr_schedule['lr_decay']['patience'],
                cooldown=self.lr_schedule['lr_decay']['cooldown'],
                eps=self.lr_schedule['lr_decay']['eps'],
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': self.lr_schedule['lr_decay']['metric_to_track']
            }

        elif self.lr_schedule['scheduler'] == 'lr_cyclic':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate)

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                               base_lr=self.lr_schedule['lr_cyclic']['lower_lr'],
                                                               max_lr=self.learning_rate,
                                                               # * self.batch_per_epoch,
                                                               step_size_up=self.lr_schedule['lr_cyclic']['epoch_size_up'],
                                                               # , * self.batch_per_epoch,
                                                               step_size_down=self.lr_schedule[
                                                                   'lr_cyclic']['epoch_size_down'],
                                                               mode=self.lr_schedule['lr_cyclic']['mode'],
                                                               cycle_momentum=False,
                                                               ),
                'interval': 'epoch',
            }

            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
            }
        # use scale fn and scale mode to overwrite mode.
        else:

            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate)

            # return {
            #     'optimizer': optimizer,
            #     # 'lr_scheduler': scheduler,
            # }

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self.get_learning_rate)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
            }
