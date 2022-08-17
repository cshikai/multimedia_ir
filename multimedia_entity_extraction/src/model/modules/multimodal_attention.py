

from typing import Dict, Tuple

import torch
import pytorch_lightning as pl
import torch.nn.functional as F


class MultimodalAttention(pl.LightningModule):
    SPLIT_VA_ACROSS_GPU = 1

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.L = cfg['model']['visual']['num_layers']
        self.M = cfg['model']['visual']['heatmap_dim']
        self.gamma_1 = cfg['model']['loss']['gamma_1']
        self.attention_negative_slope = cfg['model']['attention_negative_slope']
        self.l2_norm_eps = cfg['model']['l2_norm_eps']

    def forward(self,
                image_feature: torch.Tensor,
                word_feature: torch.Tensor,
                sentence_feature: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        word_feature dims (B,T,D)
        image_feature dims (B,M,M,L,D)
        '''
        sentence_feature = sentence_feature.unsqueeze(1)
        batch_size = word_feature.shape[0]

        word_image_heatmap = self.get_attention_heatmap(
            image_feature, word_feature, batch_size)
        # sentence_image_heatmap = self.get_attention_heatmap(
        #     image_feature, sentence_feature, batch_size)

        # word_image_score = self.get_pertinence_scores(
        #     image_feature, word_feature,  word_image_heatmap, batch_size)
        # sentence_image_score = self.get_pertinence_scores(
        #     image_feature, sentence_feature, sentence_image_heatmap, batch_size)

        # return word_image_heatmap, sentence_image_heatmap, word_image_score, sentence_image_score

        return word_image_heatmap

    def get_attention_heatmap(self, image_feature: torch.Tensor, word_feature: torch.Tensor, batch_size: int):
        max_word_len = word_feature.shape[1]

        
        # image_feature reshaped to (B,M,M,T,L,D) repeated along dim 1
        reshaped_image_feature = image_feature.unsqueeze(3).expand(-1, -1, -1, max_word_len, -1, -1)
        
        if self.SPLIT_VA_ACROSS_GPU > 1:

            # after splitting, image_feature flat dims : (B,B,M,M,T,l,D)
            reshaped_image_feature = torch.split(
                reshaped_image_feature, reshaped_image_feature.shape[5]//self.SPLIT_VA_ACROSS_GPU, dim=5)
            reshaped_image_feature = tuple([
                f.to('cuda:{:d}'.format(gpu_id)) for gpu_id, f in enumerate(reshaped_image_feature)])


            # word_feature reshaped to (B,B'M,M,T,1,D) repeated along dim 0
            reshaped_word_feature = word_feature.unsqueeze(1).unsqueeze(1).unsqueeze(
            4).unsqueeze(0).expand(batch_size, -1, self.M, self.M, -1, -1, -1)

            # heatmap dims (B,B',M,M,T,L)
            similarity_heatmap = []
            for gpu_id in range(self.SPLIT_VA_ACROSS_GPU):
                device = 'cuda:{:d}'.format(gpu_id)
                # split heatmap dims (B,B',M,M,T,l)
                similarity_heatmap_l = F.leaky_relu(F.cosine_similarity(
                    reshaped_word_feature.to(device), reshaped_image_feature[gpu_id].to(device), dim=6), negative_slope=self.attention_negative_slope)
                similarity_heatmap.append(similarity_heatmap_l)
        else:
            #originally B,1,1,T,1,D
            # word_feature reshaped to (B,B'M,M,T,L,D) repeated along dim 0
            reshaped_word_feature = word_feature.unsqueeze(1).unsqueeze(1).unsqueeze(
            4).expand(-1, self.M, self.M, -1, reshaped_image_feature.shape[4], -1)

            similarity_heatmap = F.leaky_relu(F.cosine_similarity(
                    reshaped_word_feature, reshaped_image_feature, dim=5), negative_slope=self.attention_negative_slope)
        return similarity_heatmap

    def get_pertinence_scores(self, image_feature: torch.Tensor,
                              text_feature: torch.Tensor,
                              similarity_heatmap: torch.Tensor, batch_size: int):
        max_word_len = text_feature.shape[1]

        # collapse image width and heigh dimensions into single dim for weighted summing via matrix mult
        # image_feature flat dims : (B,B',T,L,MXM,D)

        image_feature_flat = torch.flatten(image_feature, start_dim=1, end_dim=2).permute(
            0, 2, 1, 3).unsqueeze(1).unsqueeze(1).expand(-1, batch_size, max_word_len, -1, -1, -1)

        # image_feature_flat_shape = image_feature_flat.shape
        # va_shape = image_feature_flat.shape[0:3] + image_feature_flat.shape[5:]

        if self.SPLIT_VA_ACROSS_GPU > 1:
            # after splitting, image_feature flat dims : (B,B',T,l,MXM,D)
            image_feature_flat = torch.split(
                image_feature_flat, image_feature_flat.shape[3]//self.SPLIT_VA_ACROSS_GPU, dim=3)

            visual_word_attention = []
            for gpu_id in range(self.SPLIT_VA_ACROSS_GPU):
                device = 'cuda:{:d}'.format(gpu_id)
                # reshape heatmap so that dimension to sum across is at the end
                # split heatmap reshaped dims : (B,B' T, l, 1, MXM, D)
                # visual_word_attention_l dims : (B,B' T, l,D)
                visual_word_attention_l = torch.matmul(torch.flatten(
                    similarity_heatmap[gpu_id], start_dim=2, end_dim=3).permute(0, 1, 3, 4, 2).unsqueeze(4), image_feature_flat[gpu_id].to(device)).squeeze(4)

                # normalize across D means that there will be BxB'xTxl indepndant operations dividing each matrix[b,b',t,l,:] vector by its norm
                visual_word_attention_l = torch.nn.functional.normalize(
                    visual_word_attention_l, p=2, dim=-1, eps=self.l2_norm_eps)
                visual_word_attention.append(visual_word_attention_l)

            word_image_pertinence_score = []

            # text feature reshaped dims : (B,B',T,D)
            text_feature = text_feature.unsqueeze(
                1).expand(batch_size, -1, -1, -1)

            for gpu_id in range(self.SPLIT_VA_ACROSS_GPU):
                device = 'cuda:{:d}'.format(gpu_id)
                # split word_image_pertinence_score dims : (B,B',T)
                word_image_pertinence_score.append(F.cosine_similarity(
                    text_feature.to(device), visual_word_attention[gpu_id].squeeze(3).to(device), dim=3))

            # word_image_pertinence_score dims: (B, B', T,L)
            word_image_pertinence_score = torch.stack(
                [l.to('cuda') for l in word_image_pertinence_score], 3)

        else:
            
            text_feature = text_feature.unsqueeze(2).unsqueeze(
                1).expand(-1, batch_size, -1, self.L, -1)

            for gpu_id in range(self.SPLIT_VA_ACROSS_GPU):
                device = 'cuda:{:d}'.format(gpu_id)
                word_image_pertinence_score.append(F.cosine_similarity(
                    text_feature.to(device), visual_word_attention[gpu_id].squeeze(3).to(device), dim=4))

            word_image_pertinence_score = word_image_pertinence_score[0]

        # word_image_max_pertinence_score dims : (B,B',T,)
        word_image_max_pertinence_score, _ = torch.max(
            word_image_pertinence_score, dim=3)

        return word_image_max_pertinence_score
