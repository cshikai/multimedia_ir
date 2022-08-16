from typing import Dict, List,  Tuple
import math
import torch
import pytorch_lightning as pl


class TextualFeatures(pl.LightningModule):
    """
    """

    NHEAD = 8
    D_MODEL = 1024
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    ACTIVATION = 'relu'
    INPUT_DROPOUT = 0.0

    def __init__(self, cfg: Dict) -> None:
        """
        """
        super().__init__()

        self.input_dim = cfg['model']['textual']['lstm']['input_dims']
        self.hidden_dim = cfg['model']['feature_hidden_dimension']

        # assert self.hidden_dim % 2 == 0
        # self.lstm_hidden_dim = self.hidden_dim//2
        # self.num_layers = cfg['model']['textual']['lstm']['num_layers']
        # self.dropout = cfg['model']['textual']['lstm']['dropout']
        self.relu_alpha = cfg['model']['leaky_relu_alpha']
        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.D_MODEL, self.NHEAD, self.DIM_FEEDFORWARD, self.DROPOUT, self.ACTIVATION, batch_first=True)
        encoder_norm = torch.nn.LayerNorm(self.D_MODEL)
        self.transformer_encoder_1 = torch.nn.TransformerEncoder(
            encoder_layer, 1, encoder_norm)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.D_MODEL, self.NHEAD, self.DIM_FEEDFORWARD, self.DROPOUT, self.ACTIVATION, batch_first=True)
        encoder_norm = torch.nn.LayerNorm(self.D_MODEL)
        self.transformer_encoder_2 = torch.nn.TransformerEncoder(
            encoder_layer, 1, encoder_norm)

        self.positional_encoder = PositionalEncoding(
            self.D_MODEL, self.INPUT_DROPOUT)

        self.leaky_relu = torch.nn.LeakyReLU(self.relu_alpha)
        self.sentence_fc = torch.nn.Sequential(torch.nn.Linear(
            self.hidden_dim, self.hidden_dim), self.leaky_relu, torch.nn.Linear(self.hidden_dim, self.hidden_dim), self.leaky_relu)
        self.word_fc = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, self.hidden_dim), self.leaky_relu, torch.nn.Linear(
            self.hidden_dim, self.hidden_dim), self.leaky_relu)

        self.word_linear_comb = torch.nn.Linear(3, 1, bias=False)
        self.sentence_linear_comb = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor, seq_len: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        """
        source_mask = self.get_source_mask(seq_len)
        x = self.positional_encoder(x)
        output_1 = self.transformer_encoder_1(
            x, src_key_padding_mask=source_mask)
        output_2 = self.transformer_encoder_2(
            x, src_key_padding_mask=source_mask)

        word_feature = self.word_linear_comb(
            torch.stack([x, output_1, output_2, ], -1)).squeeze(-1)
        word_feature = self.word_fc(word_feature)

        first_layer_sentence_feature = output_1[:, 0, :]
        second_layer_sentence_feature = output_2[:, 0, :]
        sentence_feature = self.sentence_linear_comb(torch.stack(
            [first_layer_sentence_feature, second_layer_sentence_feature], -1)).squeeze(-1)
        sentence_feature = self.sentence_fc(sentence_feature)

        # print('word_feature', word_feature.shape)
        # print('sentence_feature', sentence_feature.shape)

        # return torch.ones(x.shape[0], x.shape[1], 1024).cuda(), torch.ones(x.shape[0], 1024).cuda()
        return word_feature, sentence_feature

        # add positional encoding to source

    def get_source_mask(self, lengths):
        return self.get_pad_mask(lengths)

    def get_pad_mask(self, lengths):

        max_len = lengths.max()
        row_vector = torch.arange(0, max_len, 1)  # size (seq_len)
        # matrix = torch.unsqueeze(lengths, dim=-1)  # size = N ,1
        mask = row_vector.to('cuda') >= lengths
        # mask = row_vector.to('cpu') >= lengths
        mask.type(torch.bool)
        return mask


class PositionalEncoding(torch.nn.Module):

    '''
    d_model : size of input embedding
    '''

    def __init__(self, d_model: int, dropout: float, max_len: int = 50):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)

        a = torch.sin(position * div_term)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), ...]
        return self.dropout(x)
