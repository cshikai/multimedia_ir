import os
import pickle
import warnings
from multiprocessing import Lock

import re
import torch

# torch.multiprocessing.set_start_method('spawn')

import transformers
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import AdamW, BertConfig, BertTokenizer

from configs import TrainConfig, TestConfig
from spacy.lang.en import English
from jerex import models, util
from jerex.task_types import TaskType
from jerex.data_module import DocREDDataModule
from jerex.entities import Document, Token, Sentence
from jerex.sampling import sampling_joint, sampling_classify

_predictions_write_lock = Lock()


class JEREXModel(pl.LightningModule):
    """ Implements the training, validation and testing routines of JEREX. """

    def __init__(self, model_type: str, tokenizer_path: str, encoder_path: str = None,
                 encoder_config_path: str = None, cache_path: str = None, lowercase: bool = False,
                 entity_types: dict = None, relation_types: dict = None,
                 prop_drop: float = 0.1,
                 meta_embedding_size: int = 25,
                 size_embeddings_count: int = 10,
                 ed_embeddings_count: int = 300,
                 token_dist_embeddings_count: int = 700,
                 sentence_dist_embeddings_count: int = 50,
                 mention_threshold: float = 0.5, coref_threshold: float = 0.5, rel_threshold: float = 0.5,
                 mention_weight: float = 1, entity_weight: float = 1, coref_weight: float = 1,
                 relation_weight: float = 1,
                 lr: float = 5e-5, lr_warmup: float = 0.1, weight_decay: float = 0.01,
                 position_embeddings_count: int = 700,
                 max_spans_train: int = None, max_spans_inference: int = None,
                 max_coref_pairs_train: int = None, max_coref_pairs_inference: int = None,
                 max_rel_pairs_train: int = None, max_rel_pairs_inference: int = None,
                 top_k_mentions_train: int = None, top_k_mentions_inference: int = None,
                 examples_filename: str = 'examples.html',
                 store_examples=True, store_predictions=True, predictions_filename='predictions.json',
                 tmp_predictions_filename='.predictions_tmp.json', **kwargs):
        super().__init__()

        self.save_hyperparameters()

        model_class = models.get_model(model_type)

        self._tokenizer = BertTokenizer.from_pretrained(tokenizer_path,
                                                        do_lower_case=lowercase,
                                                        cache_dir=cache_path)

        self._encoder_config = BertConfig.from_pretrained(encoder_config_path or encoder_path, cache_dir=cache_path)

        self.model = models.create_model(model_class, encoder_config=self._encoder_config, tokenizer=self._tokenizer,
                                         encoder_path=encoder_path, entity_types=entity_types,
                                         relation_types=relation_types,
                                         prop_drop=prop_drop, meta_embedding_size=meta_embedding_size,
                                         size_embeddings_count=size_embeddings_count,
                                         ed_embeddings_count=ed_embeddings_count,
                                         token_dist_embeddings_count=token_dist_embeddings_count,
                                         sentence_dist_embeddings_count=sentence_dist_embeddings_count,
                                         mention_threshold=mention_threshold, coref_threshold=coref_threshold,
                                         rel_threshold=rel_threshold,
                                         position_embeddings_count=position_embeddings_count,
                                         cache_path=cache_path)

        self._evaluator = model_class.EVALUATOR(entity_types, relation_types, self._tokenizer)

        task_weights = [mention_weight, coref_weight, entity_weight, relation_weight]  # loss weights of sub-components
        self._compute_loss = self.model.LOSS(task_weights=task_weights)

        self._lr = lr
        self._lr_warmup = lr_warmup
        self._weight_decay = weight_decay
        self._max_spans_train = max_spans_train
        self._max_spans_inference = max_spans_inference
        self._max_coref_pairs_train = max_coref_pairs_train
        self._max_coref_pairs_inference = max_coref_pairs_inference
        self._max_rel_pairs_train = max_rel_pairs_train
        self._max_rel_pairs_inference = max_rel_pairs_inference
        self._top_k_mentions_train = top_k_mentions_train
        self._top_k_mentions_inference = top_k_mentions_inference
        self._store_examples = store_examples
        self._store_predicitons = store_predictions

        # evaluation
        self._eval_valid_gt = None  # validation datasets converted for evaluation
        self._eval_test_gt = None  # test datasets converted for evaluation
        self._examples_filename = examples_filename
        self._predictions_filename = predictions_filename
        self._tmp_predictions_filename = tmp_predictions_filename

    def setup(self, stage):
        """ Setup is run once before training/testing starts """
        # depending on stage (training=fit or testing), convert ground truth for later evaluation
        if stage == 'fit':
            self._eval_valid_gt = self._evaluator.convert_gt(self.trainer.datamodule.valid_dataset.documents)
        elif stage == 'test':
            self._eval_test_gt = self._evaluator.convert_gt(self.trainer.datamodule.test_dataset.documents)

    def forward(self, inference=False, **batch):
        max_spans = self._max_spans_train if not inference else self._max_spans_inference
        max_coref_pairs = self._max_coref_pairs_train if not inference else self._max_coref_pairs_inference
        max_rel_pairs = self._max_rel_pairs_train if not inference else self._max_rel_pairs_inference
        top_k_mentions = self._top_k_mentions_train if not inference else self._top_k_mentions_inference


        outputs = self.model(**batch, max_spans=max_spans, max_coref_pairs=max_coref_pairs,
                             max_rel_pairs=max_rel_pairs,top_k_mentions=top_k_mentions, inference=inference)

        return outputs

    def training_step(self, batch, batch_idx):
        """ Implements a training step, i.e. calling of forward pass and loss computation """
        # this method is called by PL for every training step
        # the returned loss is optimized
        outputs = self(**batch)
        losses = self._compute_loss.compute(**outputs, **batch)
        loss = losses['loss']

        for tag, value in losses.items():
            self.log('train_%s' % tag, value.item())
        return loss

    def validation_step(self, batch, batch_idx):
        """ Implements a validation step, i.e. evaluation of validation dataset against ground truth """
        # this method is called by PL for every validation step
        # validation is run after every epoch (default)
        return self._inference(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        """ Loads current epoch's validation set predictions from disk and computes validation metrics """
        # this method is called by PL after all validation steps have finished
        if self._do_eval():
            predictions = self._load_predictions()
            metrics = self._evaluator.compute_metrics(self._eval_valid_gt[:len(predictions)], predictions)

            # this metric is used to store the best model over epochs and later use it for testing
            score = metrics[self.model.MONITOR_METRIC[0]][self.model.MONITOR_METRIC[1]]
            self.log('valid_f1', score, sync_dist=self.trainer.use_ddp, sync_dist_op='max')

            self._delete_predictions()
        else:
            self.log('valid_f1', 0, sync_dist=self.trainer.use_ddp, sync_dist_op='max')

        self._barrier()

    def test_step(self, batch, batch_idx):
        """ Implements a test step, i.e. evaluation of test dataset against ground truth """
        # return self._inference(batch, batch_idx)
        return self._inference_on_csv(batch,batch_idx)

    # def test_epoch_end(self, outputs):
    #     """ Loads current epoch's test set predictions from disk and computes test metrics """
    #     if self._do_eval():
    #         predictions = self._load_predictions()

    #         # compute evaluation metrics
    #         metrics = self._evaluator.compute_metrics(self._eval_test_gt, predictions)

    #         # log metrics
    #         for task, metrics in metrics.items():
    #             for metric_name, metric_value in metrics.items():
    #                 self.log(f'{task}_{metric_name}', metric_value)

    #         if self._store_examples:
    #             docs = self.trainer.datamodule.test_dataset.documents
    #             self._evaluator.store_examples(self._eval_test_gt, predictions, docs, self._examples_filename)

    #         if self._store_predicitons:
    #             docs = self.trainer.datamodule.test_dataset.documents
    #             self._evaluator.store_predictions(predictions, docs, self._predictions_filename)

    #         self._delete_predictions()

    #     self._barrier()

    def test_epoch_end(self, outputs):

        if os.path.isfile('./temp.csv'):
            df = pd.read_csv('./temp.csv')
            print("Results dataframe head: ", df.head())

    def _inference(self, batch, batch_index):
        """ Converts prediction results of an epoch and stores the predictions on disk for later evaluation"""
        output = self(**batch, inference=True)

        # evaluate batch
        predictions = self._evaluator.convert_batch(**output, batch=batch)

        # save predictions to disk
        with _predictions_write_lock:
            for doc_id, doc_predictions in zip(batch['doc_ids'], predictions):
                res = dict(doc_id=doc_id.item(), predictions=doc_predictions)
                with open(self._tmp_predictions_filename, 'ab+') as fp:
                    pickle.dump(res, fp)

    def _inference_on_csv(self, batch, batch_index):
        """ Converts prediction results of an epoch and stores the predictions on disk for later evaluation"""

        output = self(**batch, inference=True)

        # evaluate batch
        predictions = self._evaluator.convert_batch(**output, batch=batch)

        for doc_id, tokens,(_, mentions, clusters, entities, relations) in zip(batch['doc_ids'], batch['tokens'],predictions):

            print("Doc ID: ", doc_id.item())
            print("\n")
            print("Entities: ",entities)
            print("\n")
            print("Clusters: ",clusters)
            print("\n")
            print("Relations: ",relations)
            print("\n")

            relations_output = []
            entities_output = []

            for mention, obj_type in clusters:

                mention_spans = [list(span) for span in mention]

                entities_output.append(
                    {
                        "entity_names": [" ".join(tokens[span[0]:span[1]]) for span in mention_spans],
                        "entity_spans": mention_spans,
                        "entity_type": str(obj_type),
                    }
                )


            for sub, obj, relation_type in relations:
            
                sub_spans = list(sub[0])
                sub_type = sub[1]
                obj_spans = list(obj[0])
                obj_type = obj[1]

                

                for sub_span in sub_spans:
                    for obj_span in obj_spans:
                        relations_output.append(
                            {
                                "head": " ".join(tokens[sub_span[0]:sub_span[1]]),
                                "head_span": [sub_span[0],sub_span[1]],
                                "head_type": str(sub_type),
                                "tail": " ".join(tokens[obj_span[0]:obj_span[1]]),
                                "tail_span": [obj_span[0],obj_span[1]],
                                "tail_type": str(obj_type),
                                "relation": str(relation_type)
                            }
                        )
                        # entities_output.append(
                        #     {
                        #         "entity_name": " ".join(tokens[sub_span[0]:sub_span[1]]),
                        #         "entity_span": [sub_span[0],sub_span[1]],
                        #         "entity_type": str(sub_type),
                        #     }
                            
                        # )
                        # entities_output.append(
                        #     {
                        #         "entity_name": " ".join(tokens[obj_span[0]:obj_span[1]]),
                        #         "entity_span": [obj_span[0],obj_span[1]],
                        #         "entity_type": str(obj_type),
                        #     }
                        # )
                        
            relations_output = [i for n, i in enumerate(relations_output) if i not in relations_output[n + 1:]]
            print(relations_output)

            temp_df = pd.DataFrame(columns=['doc_id', 'tokens', 'entities','relations'])
            temp_df.loc[-1] = [doc_id.item(),tokens,entities_output,relations_output]  # adding a row
            temp_df.index = temp_df.index + 1  # shifting index
            temp_df = temp_df.sort_index()  # sorting by index

            if not os.path.isfile('./temp.csv'):
                temp_df.to_csv('./temp.csv',index=False)
            else:
                temp_df.to_csv('./temp.csv', mode='a', index=False, header=False)



    def configure_optimizers(self):
        """ Created and configures optimizer and learning rate schedule """
        # this method is called once by PL before training starts
        optimizer_params = self._get_optimizer_params()
        optimizer = AdamW(optimizer_params, lr=self._lr, weight_decay=self._weight_decay)

        dataloader = self.train_dataloader()
        train_batch_count = len(dataloader)

        gpu_dist = self.trainer.num_gpus if self.trainer.use_ddp else 1
        updates_epoch = train_batch_count // (gpu_dist * self.trainer.accumulate_grad_batches)
        updates_total = updates_epoch * self.trainer.max_epochs

        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=int(self._lr_warmup * updates_total),
                                                                 num_training_steps=updates_total)
        return [optimizer], [{'scheduler': scheduler, 'name': 'learning_rate', 'interval': 'step', 'frequency': 1}]

    def save_tokenizer(self, path):
        """ Saves tokenizer do disk """
        self._tokenizer.save_pretrained(path)

    def save_encoder_config(self, path):
        """ Saves encoder config to disk """
        self._encoder_config.save_pretrained(path)

    def _get_optimizer_params(self):
        """ Get parameters to optimize """
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self._weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _do_eval(self):
        """ Waits for all processes validation/testing end of epoch and
        decides evaluation process (with global rank 0)  """
        eval_proc = True
        self._barrier()

        if self.global_rank != 0:
            eval_proc = False

        return eval_proc

    def _barrier(self):
        """ When using ddp as accelerator, lets processes wait till all processes passed barrier """
        if self.trainer.use_ddp:
            torch.distributed.barrier(torch.distributed.group.WORLD)

    def _load_predictions(self):
        """ Load current epoch predictions from disk for evaluation """
        predictions = []
        with open(self._tmp_predictions_filename, 'rb') as fr:
            try:
                while True:
                    predictions.append(pickle.load(fr))
            except EOFError:
                pass

        predictions = sorted(predictions, key=lambda p: p['doc_id'])
        predictions = [p['predictions'] for p in predictions]
        return predictions

    def _delete_predictions(self):
        os.remove(self._tmp_predictions_filename)


def train(cfg: TrainConfig):
    """ Loads datasets, builds model and creates trainer for JEREX training"""
    if cfg.misc.seed is not None:
        pl.seed_everything(cfg.misc.seed)

    if cfg.misc.final_valid_evaluate and cfg.datasets.test_path is not None:
        warnings.warn("You set 'final_valid_evaluate=True' and specified a test path. "
                      "The best model will be evaluated on the dataset specified in 'test_path'.")

    model_class = models.get_model(cfg.model.model_type)

    tokenizer = BertTokenizer.from_pretrained(cfg.model.tokenizer_path, do_lower_case=cfg.sampling.lowercase,
                                              cache_dir=cfg.misc.cache_path)

    # read datasets
    data_module = DocREDDataModule(types_path=cfg.datasets.types_path, tokenizer=tokenizer,
                                   task_type=model_class.TASK_TYPE,
                                   train_path=cfg.datasets.train_path,
                                   valid_path=cfg.datasets.valid_path,
                                   test_path=cfg.datasets.test_path,
                                   train_batch_size=cfg.training.batch_size,
                                   valid_batch_size=cfg.inference.valid_batch_size,
                                   test_batch_size=cfg.inference.test_batch_size,
                                   neg_mention_count=cfg.sampling.neg_mention_count,
                                   neg_relation_count=cfg.sampling.neg_relation_count,
                                   neg_coref_count=cfg.sampling.neg_coref_count,
                                   max_span_size=cfg.sampling.max_span_size,
                                   neg_mention_overlap_ratio=cfg.sampling.neg_mention_overlap_ratio,
                                   final_valid_evaluate=cfg.misc.final_valid_evaluate
                                                        and cfg.datasets.test_path is None)

    data_module.setup('fit')

    model = JEREXModel(model_type=cfg.model.model_type, encoder_path=cfg.model.encoder_path,
                       tokenizer_path=cfg.model.tokenizer_path,
                       cache_path=cfg.misc.cache_path,
                       lowercase=cfg.sampling.lowercase,
                       entity_types=data_module.entity_types,
                       relation_types=data_module.relation_types,
                       prop_drop=cfg.model.prop_drop,
                       meta_embedding_size=cfg.model.meta_embedding_size,
                       size_embeddings_count=cfg.model.size_embeddings_count,
                       ed_embeddings_count=cfg.model.ed_embeddings_count,
                       token_dist_embeddings_count=cfg.model.token_dist_embeddings_count,
                       sentence_dist_embeddings_count=cfg.model.sentence_dist_embeddings_count,
                       position_embeddings_count=cfg.model.position_embeddings_count,
                       mention_threshold=cfg.model.mention_threshold, coref_threshold=cfg.model.coref_threshold,
                       rel_threshold=cfg.model.rel_threshold,
                       mention_weight=cfg.loss.mention_weight, entity_weight=cfg.loss.entity_weight,
                       coref_weight=cfg.loss.coref_weight,
                       relation_weight=cfg.loss.relation_weight,
                       lr=cfg.training.lr, lr_warmup=cfg.training.lr_warmup, weight_decay=cfg.training.weight_decay,
                       max_spans_train=cfg.training.max_spans,
                       max_spans_inference=cfg.inference.max_spans,
                       max_coref_pairs_train=cfg.training.max_coref_pairs,
                       max_coref_pairs_inference=cfg.inference.max_coref_pairs,
                       max_rel_pairs_train=cfg.training.max_rel_pairs,
                       max_rel_pairs_inference=cfg.inference.max_rel_pairs,
                       top_k_mentions_train=cfg.training.top_k_mentions,
                       top_k_mentions_inference=cfg.inference.top_k_mentions,
                       max_span_size=cfg.sampling.max_span_size)

    checkpoint_path = 'checkpoint'
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, mode='max', monitor='valid_f1')
    model.save_tokenizer(checkpoint_path)
    model.save_encoder_config(checkpoint_path)

    tb_logger = pl.loggers.TensorBoardLogger('.', 'tb')
    csv_logger = pl.loggers.CSVLogger('.', 'csv')

    trainer = pl.Trainer(callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
                         min_epochs=cfg.training.min_epochs, max_epochs=cfg.training.max_epochs,
                         logger=[tb_logger, csv_logger],
                         profiler=cfg.misc.profiler, gradient_clip_val=cfg.training.max_grad_norm,
                         gpus=cfg.distribution.gpus if cfg.distribution.gpus else None,
                         accelerator=cfg.distribution.accelerator, precision=cfg.misc.precision,
                         flush_logs_every_n_steps=cfg.misc.flush_logs_every_n_steps,
                         log_every_n_steps=cfg.misc.log_every_n_steps,
                         deterministic=cfg.misc.deterministic,
                         accumulate_grad_batches=cfg.training.accumulate_grad_batches,
                         prepare_data_per_node=cfg.distribution.prepare_data_per_node,
                         num_sanity_val_steps=0)

    trainer.fit(model, datamodule=data_module)

    # either evaluate on test_path or on valid_path if 'final_valid_evaluate=True'
    if cfg.datasets.test_path is not None or cfg.misc.final_valid_evaluate:
        # test
        data_module.setup('test')
        trainer.test(datamodule=data_module)


def test(cfg: TestConfig):
    """ Loads test dataset and model and creates trainer for JEREX testing """
    overrides = util.get_overrides_dict(mention_threshold=cfg.model.mention_threshold,
                                        coref_threshold=cfg.model.coref_threshold,
                                        rel_threshold=cfg.model.rel_threshold,
                                        cache_path=cfg.misc.cache_path)
    model = JEREXModel.load_from_checkpoint(cfg.model.model_path,
                                            tokenizer_path=cfg.model.tokenizer_path,
                                            encoder_config_path=cfg.model.encoder_config_path,
                                            max_spans_inference=cfg.inference.max_spans,
                                            max_coref_pairs_inference=cfg.inference.max_coref_pairs,
                                            max_rel_pairs_inference=cfg.inference.max_rel_pairs,
                                            top_k_mentions_inference=cfg.inference.top_k_mentions,
                                            encoder_path=None, **overrides)

    tokenizer = BertTokenizer.from_pretrained(model.hparams.tokenizer_path,
                                              do_lower_case=model.hparams.lowercase,
                                              cache_dir=model.hparams.cache_path)

    # read datasets
    model_class = models.get_model(model.hparams.model_type)
    data_module = DocREDDataModule(entity_types=model.hparams.entity_types,
                                   relation_types=model.hparams.relation_types,
                                   tokenizer=tokenizer, task_type=model_class.TASK_TYPE,
                                   test_path=cfg.dataset.test_path,
                                   test_batch_size=cfg.inference.test_batch_size,
                                   max_span_size=model.hparams.max_span_size)

    tb_logger = pl.loggers.TensorBoardLogger('.', 'tb')
    csv_logger = pl.loggers.CSVLogger('.', 'cv')

    trainer = pl.Trainer(logger=[tb_logger, csv_logger],
                         profiler="simple", gpus=cfg.distribution.gpus if cfg.distribution.gpus else None,
                         accelerator=cfg.distribution.accelerator, precision=cfg.misc.precision,
                         flush_logs_every_n_steps=cfg.misc.flush_logs_every_n_steps,
                         log_every_n_steps=cfg.misc.log_every_n_steps,
                         prepare_data_per_node=cfg.distribution.prepare_data_per_node)

    # test
    data_module.setup('test')
    trainer.test(model, datamodule=data_module)

def test_on_df(cfg: TestConfig):
    """ Loads test dataset and model and creates trainer for JEREX testing """
    overrides = util.get_overrides_dict(mention_threshold=cfg.model.mention_threshold,
                                        coref_threshold=cfg.model.coref_threshold,
                                        rel_threshold=cfg.model.rel_threshold,
                                        cache_path=cfg.misc.cache_path)
    model = JEREXModel.load_from_checkpoint(cfg.model.model_path,
                                            tokenizer_path=cfg.model.tokenizer_path,
                                            encoder_config_path=cfg.model.encoder_config_path,
                                            max_spans_inference=cfg.inference.max_spans,
                                            max_coref_pairs_inference=cfg.inference.max_coref_pairs,
                                            max_rel_pairs_inference=cfg.inference.max_rel_pairs,
                                            top_k_mentions_inference=cfg.inference.top_k_mentions,
                                            encoder_path=None, **overrides)

    tokenizer = BertTokenizer.from_pretrained(model.hparams.tokenizer_path,
                                              do_lower_case=model.hparams.lowercase,
                                              cache_dir=model.hparams.cache_path)

    # read datasets
    model_class = models.get_model(model.hparams.model_type)
    data_module = DocREDDataModule(entity_types=model.hparams.entity_types,
                                   relation_types=model.hparams.relation_types,
                                   tokenizer=tokenizer, task_type=model_class.TASK_TYPE,
                                   test_path=cfg.dataset.test_path,
                                   test_batch_size=cfg.inference.test_batch_size,
                                   max_span_size=model.hparams.max_span_size,sampling_processes=cfg.sampling.sampling_processes)

    tb_logger = pl.loggers.TensorBoardLogger('.', 'tb')
    csv_logger = pl.loggers.CSVLogger('.', 'cv')

    trainer = pl.Trainer(logger=[tb_logger, csv_logger],
                         profiler="simple", gpus=cfg.distribution.gpus if cfg.distribution.gpus else None,
                         accelerator=cfg.distribution.accelerator, precision=cfg.misc.precision,
                         flush_logs_every_n_steps=cfg.misc.flush_logs_every_n_steps,
                         log_every_n_steps=cfg.misc.log_every_n_steps,
                         prepare_data_per_node=cfg.distribution.prepare_data_per_node)

    # test
    data_module.setup('test')
    trainer.test(model, datamodule=data_module)

    if os.path.isfile('./temp.csv'):
            df = pd.read_csv('./temp.csv')
            os.remove('./temp.csv')
            return df
    else:
        return None
    
def parse_sentences(tokenizer, jsentences):

    tid = 0
    sid = 0
    sentences = []

    # full document sub-word encoding
    doc_encoding = []

    # parse tokens
    tok_doc_idx = 0
    for sidx, jtokens in enumerate(jsentences):
        sentence_tokens = []

        for tok_sent_idx, token_phrase in enumerate(jtokens):
            token_encoding = tokenizer.encode(token_phrase, add_special_tokens=False, truncation=True)
            if not token_encoding:
                token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

            token = Token(tid, tok_doc_idx, tok_sent_idx, span_start, span_end, token_phrase)
            tid += 1

            sentence_tokens.append(token)
            doc_encoding += token_encoding

            tok_doc_idx += 1

        sentence = Sentence(sid, sidx, sentence_tokens)
        sid += 1

        sentences.append(sentence)

    return sentences, doc_encoding


def inference_on_fly(model, tokenizer, paras, task):
    """ For running inference on one sample passed in, on-the-fly either through command line or API call"""

    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    relation_df = pd.DataFrame(columns=['doc_id', 'tokens', 'relations'])

    for doc_idx, para in enumerate(paras):
        print(para)

        str_sents = list(nlp(para).sents)
        token_sents = []
        relations_output = []
        num_tokens = 0
        for sent in str_sents:
            tokens = list(nlp.tokenizer(sent.text))
            tokens = [token.text.strip() for token in tokens]
            if num_tokens + len(tokens) > 500:
                break
            num_tokens += len(tokens)
            token_sents.append(tokens)

        sentences, doc_encoding = parse_sentences(tokenizer, token_sents)
        doc = Document(doc_idx, util.flatten(token_sents), sentences, [], [], doc_encoding, 'generic')

        # doc = Document(doc_id=idx, tokens=[0,], sentences: List[Sentence],
        #          entities: List[Entity], relations: List[Relation], encoding: List[int], title: str)

        if task == TaskType.JOINT:
            samples = sampling_joint.create_joint_inference_sample(doc, model.hparams.max_span_size)
        elif task == TaskType.MENTION_LOCALIZATION:
            samples = sampling_classify.create_mention_classify_inference_sample(doc, model.hparams.max_span_size)
        elif task == TaskType.COREFERENCE_RESOLUTION:
            samples = sampling_classify.create_coref_classify_inference_sample(doc)
        elif task == TaskType.ENTITY_CLASSIFICATION:
            samples = sampling_classify.create_entity_classify_sample(doc)
        elif task == TaskType.RELATION_CLASSIFICATION:
            samples = sampling_classify.create_rel_classify_inference_sample(doc)
        else:
            raise Exception('Invalid task')

        samples['doc_ids'] = torch.tensor(doc.doc_id, dtype=torch.long)
        padded_batch = dict()
        batch = [samples,]


        for key in samples.keys():
            samples = [s[key] for s in batch]

            if not batch[0][key].shape:
                padded_batch[key] = torch.stack(samples)
            else:
                padded_batch[key] = util.padded_stack([s[key] for s in batch])

        with torch.no_grad():
            output = model(**padded_batch, inference=True)

        # evaluate batch
        _, mentions, clusters, entities, relations = model._evaluator.convert_batch(**output, batch=padded_batch)[0]

        print("Entities: ",entities)
        print("Relations: ",relations)

        tokens = [item for sublist in token_sents for item in sublist]


        for sub, obj, relation_type in relations:
            
            sub_spans = list(sub[0])
            sub_type = sub[1]
            obj_spans = list(obj[0])
            obj_type = obj[1]

            for sub_span in sub_spans:
                for obj_span in obj_spans:
                    relations_output.append(
                        {
                            "head": " ".join(tokens[sub_span[0]:sub_span[1]]),
                            "head_span": [sub_span[0],sub_span[1]],
                            "head_type": str(sub_type),
                            "tail": " ".join(tokens[obj_span[0]:obj_span[1]]),
                            "tail_span": [obj_span[0],obj_span[1]],
                            "tail_type": str(obj_type),
                            "relation": str(relation_type)
                        }
                    )

        
        relation_df.loc[-1] = [doc_idx,tokens,relations_output]  # adding a row
        relation_df.index = relation_df.index + 1  # shifting index
        relation_df = relation_df.sort_index()  # sorting by index

    # relation_df = relation_df.loc[relation_df.astype(str).drop_duplicates().index]

    return relation_df.reset_index(drop=True)

def api_call_single(cfg: TestConfig, docs):
    """ Loads test dataset and model and creates trainer for JEREX testing """
    overrides = util.get_overrides_dict(mention_threshold=cfg.model.mention_threshold,
                                        coref_threshold=cfg.model.coref_threshold,
                                        rel_threshold=cfg.model.rel_threshold,
                                        cache_path=cfg.misc.cache_path)
    model = JEREXModel.load_from_checkpoint(cfg.model.model_path,
                                            tokenizer_path=cfg.model.tokenizer_path,
                                            encoder_config_path=cfg.model.encoder_config_path,
                                            max_spans_inference=cfg.inference.max_spans,
                                            max_coref_pairs_inference=cfg.inference.max_coref_pairs,
                                            max_rel_pairs_inference=cfg.inference.max_rel_pairs,
                                            top_k_mentions_inference=cfg.inference.top_k_mentions,
                                            encoder_path=None, **overrides).eval()


    tokenizer = BertTokenizer.from_pretrained(model.hparams.tokenizer_path,
                                              do_lower_case=model.hparams.lowercase,
                                            #   do_lower_case=True,
                                              cache_dir=model.hparams.cache_path)

    # read datasets
    model_class = models.get_model(model.hparams.model_type)

    relation_df = inference_on_fly(model, tokenizer, docs,'joint')

    return relation_df

def test_on_fly(cfg: TestConfig):
    """ Loads test dataset and model and creates trainer for JEREX testing """
    overrides = util.get_overrides_dict(mention_threshold=cfg.model.mention_threshold,
                                        coref_threshold=cfg.model.coref_threshold,
                                        rel_threshold=cfg.model.rel_threshold,
                                        cache_path=cfg.misc.cache_path)
    model = JEREXModel.load_from_checkpoint(cfg.model.model_path,
                                            tokenizer_path=cfg.model.tokenizer_path,
                                            encoder_config_path=cfg.model.encoder_config_path,
                                            max_spans_inference=cfg.inference.max_spans,
                                            max_coref_pairs_inference=cfg.inference.max_coref_pairs,
                                            max_rel_pairs_inference=cfg.inference.max_rel_pairs,
                                            top_k_mentions_inference=cfg.inference.top_k_mentions,
                                            encoder_path=None, **overrides).eval()


    tokenizer = BertTokenizer.from_pretrained(model.hparams.tokenizer_path,
                                              do_lower_case=model.hparams.lowercase,
                                            #   do_lower_case=True,
                                              cache_dir=model.hparams.cache_path)

    # read datasets
    model_class = models.get_model(model.hparams.model_type)

    df = pd.read_csv(cfg.dataset.test_path)

    lowercased = 'The three-night cruises will stop at Penang, while the four-night cruises will stop at both Penang and Port Klang near Kuala Lumpur, with a range of shore excursions available to guests in the two ports of call. These include visits to Penang’s St George’s Church and Batu Caves on the outskirts of the Malaysian capital.'
    uppercased = 'The announcement ends more than two years of cruises-to-nowhere, which came as operators attempt to make cruising safer amid the COVID-19 pandemic. Guests are required to have six months\' validity on their passports and must update the MySejahtera app before they depart. They are also required to comply with local vaccination requirements. Bookings for the cruises opened on Thursday. “We are thrilled to be the first cruise line in Singapore to reconnect holidaymakers with Asia’s beautiful destinations once again,” said Royal Caribbean\'s Asia-Pacific vice president and managing director Angie Stephen. “The vibrant and culture-rich cities of Penang and Kuala Lumpur have so much to offer, and that is only the beginning.” Singapore Tourism Board\'s director of cruise Annie Chang said that holidaymakers can look forward to more international cruises in the near future. "We have been working closely with various governments in southeast asia to align on cruise protocols and policies, and are excited to bring back port calls in Malaysia for sailings as a start,” she said. “Port calls will provide more vacation options and we look forward to seeing more first-time and repeat cruisers in the coming year as more ports in the region open up.”'

    docs = [lowercased+uppercased,]

    relation_df = inference_on_fly(model, tokenizer, docs,'joint')
    relation_df.to_csv('output.csv')


    return relation_df
