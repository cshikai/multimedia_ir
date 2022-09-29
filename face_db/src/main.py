from http import server
import os
from typing import List

import torch
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

api = FastAPI(
    title='Image Server API',
    description='API for Image Server. Standin for a S3 for prototyping purposes',
    version='1.0.0'
)


class M2E2EmbMgr:
    def __init__(self):

        self.emb_root = '../embeddings/M2E2'
        # {"n.pt":{"file_name":"image_name", "index":m}
        self.map_file = '../embeddings/M2E2_map.json'
        self.mapping = None
        self.emb_len = None
        self.all_emb = None
        self._refresh()

    def _refresh(self):  # Call upon upload
        self.emb_len = len(os.listdir(self.emb_root))
        self.mapping = self._load_mapping()
        self.all_emb = self._load_emb()

    def _load_emb(self):
        emb_list = []
        emb_files = os.listdir(self.emb_root)
        for file in emb_files:
            file_emb = torch.load(f"{self.emb_root}/{file}")
            emb_list.append(file_emb)
        if len(emb_list) == 0:
            all_emb = []
        else:
            all_emb = torch.cat(emb_list)
        return all_emb

    def _load_mapping(self):
        with open(self.map_file, 'r') as f:
            map_dict = json.load(f)
        return map_dict

    def upload(self, new_emb, file_name, img_index):
        self.emb_len = len(os.listdir(self.emb_root))
        emb_file = f"{str(self.emb_len)}.pt"
        emb_tensor = torch.Tensor(new_emb)
        emb_tensor = emb_tensor/torch.linalg.norm(emb_tensor)  # Normalisation
        emb_tensor = torch.unsqueeze(emb_tensor, 0)
        if self.emb_len == 0:
            self.all_emb = emb_tensor
        else:
            print(self.all_emb.shape, emb_tensor.shape)
            self.all_emb = torch.cat([self.all_emb, emb_tensor])
        torch.save(emb_tensor, f"{self.emb_root}/{emb_file}")
        self.mapping[emb_file] = {
            'file_name': file_name,
            'index': img_index
        }
        self.emb_len += 1
        with open(self.map_file, 'w') as f:
            json.dump(self.mapping, f,  indent=4)
        return "Success"

    def compare(self, source_emb, k):
        emb_tensor = torch.Tensor(source_emb)
        emb_tensor = emb_tensor/torch.linalg.norm(emb_tensor)  # Normalisation
        norm_emb = torch.transpose(self.all_emb, 0, 1)
        print(emb_tensor.size(), norm_emb.size())
        cos_sim = torch.matmul(emb_tensor, norm_emb)
        print(cos_sim)
        topk_cos = torch.topk(cos_sim, k)
        conf = topk_cos.values.tolist()
        topk_index = topk_cos.indices.tolist()
        print(topk_cos, topk_index)
        similar_entity_files = [self.mapping[f"{i}.pt"] for i in topk_index]

        return similar_entity_files, conf


EmbMgr = M2E2EmbMgr()


class PopulateFaceEmb(BaseModel):
    face: List[float]
    file_name: str
    index: int


class SourceFaceEmb(BaseModel):
    face: List[float]
    k: int


@ api.put("/upload/")
def upload_emb(image_data: PopulateFaceEmb):
    image_data = image_data.dict()

    response = {}
    # try:
    response['status'] = EmbMgr.upload(
        image_data['face'], image_data['file_name'], image_data['index'])
    # except:
    #     response['status'] = 'failed'

    return response


@ api.get("/compare/")
def compare_emb(image_data: SourceFaceEmb):
    image_data = image_data.dict()
    face_emb = image_data['face']
    k = image_data['k']

    entities, conf = EmbMgr.compare(face_emb, 5)  # Top 5 results

    response = {}
    response['files'] = entities
    response['similarity'] = conf
    return response
