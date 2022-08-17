
import os
from typing import Dict


import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader


from config.config import cfg
from .visual_model import VGG
from .dataset import VGImageDataset
from .preprocessor import PreProcessor


class EmbeddingGenerator():
    BATCH_SIZE = 256
    REPORT_INTERVAL = 1
   # should init as arguments here

    def __init__(self, cfg: Dict, clearml_task=None) -> None:
        self.clearml_task = clearml_task
        self.cfg = cfg

    def run(self, dataset_root) -> None:
        # if os.path.exists(self.checkpoint_dir):
        #     shutil.rmtree(self.checkpoint_dir)

        # os.makedirs(os.path.join(self.checkpoint_dir,'logs'), exist_ok=True)
        pl.seed_everything(self.cfg['training']['seed'])

        self._set_datasets()

        self._set_dataloaders()

        self._initialize_model()

        self._generate(dataset_root)

    def _initialize_model(self) -> None:
        self.model = VGG(self.cfg)

    def _set_datasets(self) -> None:
        self.train_dataset = VGImageDataset('train', cfg)
        self.valid_dataset = VGImageDataset('valid', cfg)

    def _set_dataloaders(self) -> None:
        # self.cfg['training']['batch_size']
        self.train_loader = DataLoader(self.train_dataset, collate_fn=self.collate,
                                       batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.cfg['training']['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset, collate_fn=self.collate,
                                       batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.cfg['training']['num_workers'])

    def _generate(self, dataset_root):

        folders = ['train', 'valid']
        sub_folders = ['flickr', 'mscoco', 'visualgenome']

        self.model.cuda()
        self.model.eval()

        print('vgg embedder started on gpu?:',
              next(self.model.parameters()).is_cuda)

        for f in folders:
            for sub_f in sub_folders:
                path = os.path.join(dataset_root, f, 'image', sub_f)

                if not os.path.exists(path):
                    os.makedirs(path)

        processed = 0
        train_len = len(self.train_dataset)
        for batch in self.train_loader:
            processed += 1
            batch_image, batch_name = batch

            output = self.model(batch_image.cuda())

            if processed % self.REPORT_INTERVAL == 0:
                print('processing train image {}/{} '.format(processed *
                      self.BATCH_SIZE, train_len))

            for number, name in enumerate(batch_name):
                single = output[number, ...].clone()

                torch.save(
                    single, os.path.join(
                        dataset_root, 'train/image/{}'.format(name)))

        processed = 0
        valid_len = len(self.valid_dataset)
        for batch in self.valid_loader:
            processed += 1
            batch_image, batch_name = batch

            output = self.model(batch_image.cuda())

            if processed % self.REPORT_INTERVAL == 0:
                print('processing valid image {}/{} '.format(processed *
                      self.BATCH_SIZE, valid_len))

            for number, name in enumerate(batch_name):
                single = output[number, ...].clone()
                torch.save(
                    single, os.path.join(
                        dataset_root, 'valid/image/{}'.format(name)))

    @ staticmethod
    def create_torchscript_model(model_name: str) -> None:
        model = VGG.load_from_checkpoint(model_name)
        model.eval()
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(
            '/ models/trained_models /', "model.pt"))

    def collate(self, batch):
        # sort by len
        # batch.sort(key=lambda x: x[-1], reverse=True)
        batch_image, batch_index, = zip(*batch)
        # batch_pad_text = torch.nn.utils.rnn.pad_sequence(
        #     batch_text, batch_first=True, padding_value=0)

        batch_image = torch.stack(batch_image, 0)

        return batch_image, batch_index
