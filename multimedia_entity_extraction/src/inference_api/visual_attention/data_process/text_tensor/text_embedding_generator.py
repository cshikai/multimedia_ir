
import os
from typing import Dict


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from clearml import StorageManager


from config.config import cfg
from .textual_model import Elmo
from .dataset import VGTextDataset
from .preprocessor import PreProcessor


class EmbeddingGenerator():
    REPORT_INTERVAL = 1
    BATCH_SIZE = 256*2
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
        self.model = Elmo(self.cfg)

    def _set_datasets(self) -> None:

        self.train_dataset = VGTextDataset('train', cfg)
        self.valid_dataset = VGTextDataset('valid', cfg)

    def _set_dataloaders(self) -> None:
        # self.cfg['training']['batch_size']
        self.train_loader = DataLoader(self.train_dataset, collate_fn=self.train_dataset.preprocessor.collate,
                                       batch_size=self.BATCH_SIZE, shuffle=False,
                                       num_workers=self.cfg['training']['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset, collate_fn=self.valid_dataset.preprocessor.collate,
                                       batch_size=self.BATCH_SIZE, shuffle=False,
                                       num_workers=self.cfg['training']['num_workers'])

    def _generate(self, dataset_root):

        folders = ['train', 'valid']

        self.model.cuda()
        self.model.eval()
        print('elmo embedder started on gpu?:',
              next(self.model.parameters()).is_cuda)

        for f in folders:
            path = os.path.join(dataset_root, f, 'text')

            if not os.path.exists(path):
                os.makedirs(path)
        train_len = len(self.train_dataset)

        for batch in self.train_loader:
            print('Starting batch...')
            batch_text, batch_index, batch_len = batch
            index = min(batch_index)
            output = self.model(batch_text.cuda())

            if index % self.REPORT_INTERVAL == 0:
                print('processing train text {}/{} '.format(index+1, train_len))

            for number, idx in enumerate(batch_index):
                single = output[number, :batch_len[number], ...].clone()

                torch.save(
                    single, os.path.join(dataset_root, 'train/text/{}'.format(idx)))
            print('saved embeddings to file system')

        valid_len = len(self.valid_dataset)
        for batch in self.valid_loader:
            print('Starting batch...')
            batch_text, batch_index, batch_len = batch
            output = self.model(batch_text.cuda())
            index = min(batch_index)
            if index % self.REPORT_INTERVAL == 0:
                print('processing valid text {}/{} '.format(index+1, valid_len))

            for number, idx in enumerate(batch_index):
                single = output[number, :batch_len[number], ...].clone()

                torch.save(
                    single, os.path.join(dataset_root, 'valid/text/{}'.format(idx)))
            print('saved embeddings to file system')

    @staticmethod
    def create_torchscript_model(model_name: str) -> None:
        model = Elmo.load_from_checkpoint(model_name)
        model.eval()
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(
            '/ models/trained_models /', "model.pt"))
