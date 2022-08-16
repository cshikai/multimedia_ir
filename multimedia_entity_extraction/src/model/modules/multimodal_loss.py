

from typing import List

import torch


class MultimodalLoss(torch.nn.Module):

    def __init__(self, gamma_1: int, gamma_2: int) -> None:
        super().__init__()

        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    def forward(self,
                word_image_max_score: torch.Tensor,
                sentence_image_max_score: torch.Tensor,
                seq_lens: List[int]
                ) -> torch.Tensor:
        '''
        word_image_max_score dims (B,B',T)
        sentence_image_max_score dims (B,B',1)
        seq_lens dims (B,1)        
        '''
        batch_size = word_image_max_score.shape[0]

        aggregated_sentence_image_score = torch.exp(
            word_image_max_score * self.gamma_1) * self._get_len_mask(batch_size, seq_lens[0], seq_lens)

        # aggregated_sentence_image_score dims: (B,B')
        aggregated_sentence_image_score = torch.log(
            torch.sum(aggregated_sentence_image_score, 2) ** (1/self.gamma_1))

        # sentence_image_score dims: (B,B')
        sentence_image_score = sentence_image_max_score.squeeze(2)

        assert sentence_image_score.shape[0] == batch_size and sentence_image_score.shape[1] == batch_size and len(
            sentence_image_score.shape) == 2
        return self._get_multimodal_loss(aggregated_sentence_image_score) + self._get_multimodal_loss(sentence_image_score)

    def _get_multimodal_loss(self, sentence_image_score):

        score = torch.exp(sentence_image_score * self.gamma_2)
        fixed_image_score = score / torch.sum(score, dim=1, keepdim=True)
        fixed_sentence_score = score / torch.sum(score, dim=0, keepdim=True)

        # print('probs', torch.sum(score, dim=1, keepdim=True))
        # print(torch.diagonal(fixed_image_score, 0))
        # print(torch.diagonal(fixed_sentence_score, 0))

        loss = -torch.sum(torch.log(torch.diagonal(fixed_image_score, 0)) +
                          torch.log(torch.diagonal(fixed_sentence_score, 0)))
        del fixed_image_score
        del fixed_sentence_score
        torch.cuda.empty_cache()
        return loss

    def _get_len_mask(self, batch_size, max_len, seq_lens):
        """Generates an 'upper-triangular matrix' with 1 in places without mask"""
        block = torch.zeros(batch_size, max_len, device='cuda')
        for i, l in enumerate(seq_lens):
            block[i, :l] = torch.ones(1, l)
        block = block.unsqueeze(0).expand(batch_size, -1, -1)
        return block
