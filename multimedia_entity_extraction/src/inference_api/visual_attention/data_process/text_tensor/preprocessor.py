import os
from typing import Dict, Tuple
import torch
import numpy as np

from PIL.Image import Image
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.fields.text_field import TextField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from .. import transforms


class PreProcessor():

    def __init__(self, cfg: Dict) -> None:
        self.indexer = ELMoTokenCharactersIndexer()
        self.vocab = Vocabulary()
        self.tokenizer = WhitespaceTokenizer()
        self.text_field = TextField(self.tokenizer.tokenize('test'), {
                                    "elmo_tokens": self.indexer})
        self.text_field.index(self.vocab)
        self.cfg = cfg

        self.text_transforms = transforms.get_transforms(
            self.cfg['data']['transforms']['text'])

    def __call__(self, text:  str) -> torch.Tensor:
        

        text = self.text_transforms(text)  # .to('cuda')

        return text

    def collate(self, batch):
        # sort by len
  


        # batch.sort(key=lambda x: x[-1], reverse=True)
        batch_text, batch_index, batch_len, _ ,_ , _= zip(*batch)
        # batch_pad_text = torch.nn.utils.rnn.pad_sequence(
        #     batch_text, batch_first=True, padding_value=0)

        batch_text = torch.nn.utils.rnn.pad_sequence(
            batch_text, batch_first=True, padding_value=261)

        # # batch_text = torch.stack(batch_text, 0)
        # tensor_dict = self.text_field.batch_tensors(batch_text)

        return batch_text, batch_index, batch_len
