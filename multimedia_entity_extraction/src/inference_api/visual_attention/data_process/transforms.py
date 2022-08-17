

import numpy as np
from typing import List


from torchvision.transforms import Compose
import torch
import torchvision
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields.text_field import TextField
from allennlp.data.tokenizers.token_class import Token
from PIL.Image import Image


class UpperClamp():
    '''
    Clamps the values per column
    '''

    def __init__(self, upper_bound: List[float]) -> None:
        self.upper_bound = torch.Tensor(upper_bound).unsqueeze(0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.min(x, self.upper_bound)


class LowerClamp():
    '''
    Clamps the values per column
    '''

    def __init__(self, lower_bound: List[float]) -> None:
        self.lower_bound = torch.Tensor(lower_bound).unsqueeze(0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, self.lower_bound)


class Normalize():
    '''
    Normalizes each feature across time.
    output will be between -1,1
    (sequence,features)
    '''

    def __init__(self, kwargs) -> None:
        '''
        norm_values has shape (1,num_features)
        '''

        upper_norm_values = kwargs['upper_norm_values']
        lower_norm_values = kwargs['lower_norm_values']

        upper_norm_values = np.expand_dims(np.array(upper_norm_values), axis=0)
        self.lower_norm_values = np.expand_dims(
            np.array(lower_norm_values), axis=0)
        self.range = upper_norm_values - self.lower_norm_values

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input : (sequence,features)
        output : (sequence,features)
        '''
        x = x - self.lower_norm_values
        return (x/self.range)*2 - 1


class NormalizeZeroOne():
    '''
    Normalizes each feature across time.
    output will be between 0,1
    (sequence,features)
    '''

    def __init__(self, **kwargs) -> None:
        '''
        norm_values has shape (1,num_features)
        '''
        self.upper_norm_values = np.expand_dims(
            np.array(kwargs['upper_norm_values']), axis=(1, 2))
        self.lower_norm_values = np.expand_dims(
            np.array(kwargs['lower_norm_values']), axis=(1, 2))
        self.range = self.upper_norm_values - self.lower_norm_values

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input : (sequence,features)
        output : (sequence,features)
        '''
        x = x - self.lower_norm_values
        return (x/self.range)


class ImageResize():

    def __init__(self, **kwargs):

        if 'size' in kwargs.keys():
            self.resizer = torchvision.transforms.Resize(kwargs['size'])
        else:
            height = kwargs['height']
            width = kwargs['width']
            self.resizer = torchvision.transforms.Resize((height, width))

    def __call__(self, x: Image) -> Image:
        return self.resizer(x)


class NormalizeMeanStd():

    def __init__(self, **kwargs):
        mean = kwargs['mean']
        std = kwargs['std']
        self.normalizer = torchvision.transforms.Normalize(mean, std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalizer(x)


class ToTensor():
    '''
    Does 3 operations:

    1. Maps PIL image to tensor.

    2. Transform the shape from (h,w,c) to (c,h,w)

    3. Normalize all values to between 0 and 1
    '''

    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, x: Image) -> torch.Tensor:
        return self.to_tensor(x)


class CenterCrop():

    def __init__(self, **kwargs):
        size = kwargs['size']
        self.crop = torchvision.transforms.CenterCrop(size)

    def __call__(self, x: Image) -> Image:
        return self.crop(x)


class Word2Char():

    def __init__(self,):
        self.tokenizer = WhitespaceTokenizer()

    def __call__(self, x: str) -> List[Token]:
        return self.tokenizer.tokenize(x)


class ElmoChar2Index():

    def __init__(self, ):
        '''
        As elmo indexer already comes with prebuilt vocab, we initialize empty vocabulary object, 
        '''
        self.indexer = ELMoTokenCharactersIndexer()
        self.vocab = Vocabulary()

    def __call__(self, x:  List[Token]) -> torch.Tensor:
        text_field = TextField(x, {"elmo_tokens": self.indexer})
        text_field.index(self.vocab)
        token_tensor = text_field.as_tensor(
            text_field.get_padding_lengths())['elmo_tokens']['elmo_tokens']
        # tensor_dict = text_field.batch_tensors([token_tensor])
        return token_tensor


TRANSFORM_MAPPER = {

    'UpperClamp': UpperClamp,
    'LowerClamp': LowerClamp,
    'Normalize': Normalize,
    'NormalizeZeroOne': NormalizeZeroOne,
    'Resize': ImageResize,
    'CenterCrop': CenterCrop,
    'NormalizeMeanStd': NormalizeMeanStd,
    'ToTensor': ToTensor,
    'Word2Char': Word2Char,
    'ElmoChar2Index': ElmoChar2Index,
    'ToTensor': ToTensor,



}


def get_transforms(transforms_dict):
    x_transforms = []
    for k, kwargs in transforms_dict.items():

        if kwargs != None:
            x_transforms.append(TRANSFORM_MAPPER[k](**kwargs))
        else:
            x_transforms.append(TRANSFORM_MAPPER[k]())
    return Compose(x_transforms)
