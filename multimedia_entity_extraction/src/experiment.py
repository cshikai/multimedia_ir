
import os
from typing import Dict


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from clearml import StorageManager


from config.config import cfg
from model.model import VisualGroundingModel
from data.dataset import VisualGroundingDataset
from data.preprocessor import PreProcessor


class Experiment():

   # should init as arguments here
    def __init__(self, cfg: Dict, clearml_task=None) -> None:
        self.clearml_task = clearml_task
        self.cfg = cfg

    def run_experiment(self) -> None:
        # if os.path.exists(self.checkpoint_dir):
        #     shutil.rmtree(self.checkpoint_dir)

        # os.makedirs(os.path.join(self.checkpoint_dir,'logs'), exist_ok=True)
        pl.seed_everything(self.cfg['training']['seed'])
        self.distributed = cfg['training']['n_gpu'] > 1

        self._set_datasets()

        self._set_dataloaders()

        self._log_metadata()

        # load from checkpoint
        if self.cfg['training']['resume_from_checkpoint']:
            self._load_model()
        else:
            self._initialize_model()
        self._set_callbacks()

        self._set_trainer()

        self._start_training()

    def _start_training(self) -> None:

        if self.cfg['training']['auto_lr']:
            lr_finder = self.trainer.tuner.lr_find(
                self.model, self.train_loader, self.valid_loader)
            new_lr = lr_finder.suggestion()
            self.model.learning_rate = new_lr
            self.print('Found a starting LR of {}'.format(new_lr))
        self.trainer.fit(self.model, self.train_loader, self.valid_loader)

    def _set_trainer(self) -> None:
        self.trainer = pl.Trainer(
            devices=self.cfg['training']['n_gpu'],
            accelerator="gpu",
            # strategy=self.cfg['training']['accelerator'] if self.distributed else None,
            strategy=self.cfg['training']['accelerator'],
            accumulate_grad_batches=self.cfg['training']['accumulate_grad_batches'],
            callbacks=self.callbacks,
            logger=self._get_logger(),
            max_epochs=self.cfg['training']['epochs'],
            default_root_dir=self.cfg['training']['local_trained_model_path'],
            log_every_n_steps=self.cfg['training']['log_every_n_steps']
        )

    def _log_metadata(self) -> None:
        pass
        # class_weights = {
        #     'label_segment_count': train_dataset.get_class_weights('label_segment_count'),
        #     'label_point_count': train_dataset.get_class_weights('label_point_count'),
        #     'None': train_dataset.get_class_weights('None')
        # }
        # labels_map = train_dataset.labels_map
        # n_callsign_tokens = len(train_dataset.CALLSIGN_CHAR2IDX)
        # n_mode3_tokens = len(train_dataset.MODE3_CHAR2IDX)
        # n_classes = train_dataset.n_classes
        # distributed = self.n_gpu > 1
        # if self.clearml_task:
        #     if self.weight_by != 'None':
        #         self.clearml_task.connect_configuration({str(i): val for i, val in enumerate(
        #             class_weights[self.weight_by].cpu().numpy())}, name='Class Weights')
        #     self.clearml_task.connect_configuration(
        #         labels_map, name='Labels Map')

        # metas = {'Train': train_dataset.metadata.copy(
        #       ), 'Valid': valid_dataset.metadata.copy()}
        #   for meta in metas.keys():
        #        for key in ['labels', 'length', 'track_ids']:
        #             metas[meta].pop(key)
        #         self.clearml_task.connect_configuration(
        #             metas[meta], name='{} Metadata'.format(meta))

    def _initialize_model(self) -> None:
        self.model = VisualGroundingModel(self.cfg, self.distributed)

    def _load_model(self) -> None:

        if self.task:
            local_trained_model_path = StorageManager.get_local_copy(
                cfg['training']['trained_model_id'])
        else:
            local_trained_model_path = os.path.join(
                cfg['training']['local_trained_model_path'], 'latest_model.ckpt')
        self.model = VisualGroundingModel.load_from_checkpoint(
            local_trained_model_path, self.cfg, self.distributed)

    def _set_datasets(self) -> None:
        preprocessor = PreProcessor(cfg)
        self.train_dataset = VisualGroundingDataset('train', preprocessor)
        self.valid_dataset = VisualGroundingDataset('valid', preprocessor)

        # self.train_dataset = VisualGroundingDataset('train', cfg)
        # self.valid_dataset = VisualGroundingDataset('valid', cfg)

    def _set_dataloaders(self) -> None:

        self.train_loader = DataLoader(self.train_dataset, collate_fn=self.train_dataset.preprocessor.collate,
                                       batch_size=self.cfg['training']['batch_size'], shuffle=False, num_workers=self.cfg['training']['num_workers'])
        self.valid_loader = DataLoader(self.valid_dataset, collate_fn=self.valid_dataset.preprocessor.collate,
                                       batch_size=self.cfg['training']['batch_size'], shuffle=False, num_workers=self.cfg['training']['num_workers'])

    def _get_logger(self) -> TensorBoardLogger:
        logger = TensorBoardLogger(
            cfg['training']['local_trained_model_path'], name='logs')
        return logger

    def _set_callbacks(self) -> None:
        self.callbacks = []

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.cfg['training']['local_trained_model_path'],
            filename='best',
            save_top_k=1,
            verbose=True,
            save_last=True,
            monitor='val_loss',
            mode='min',
            every_n_epochs=self.cfg['training']['model_save_period']
        )
        self.callbacks.append(checkpoint_callback)
        if self.cfg['training']['lr_schedule']['scheduler']:
            lr_logging_callback = LearningRateMonitor(logging_interval='step')
            self.callbacks.append(lr_logging_callback)

    @staticmethod
    def create_torchscript_model(model_name: str) -> None:
        model = VisualGroundingDataset.load_from_checkpoint(model_name)
        model.eval()
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(
            '/ models/trained_models /', "model.pt"))
