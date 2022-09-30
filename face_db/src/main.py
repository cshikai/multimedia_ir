import os
from typing import List

import torch
import json
import re
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
        self.reverse_map = None
        self._refresh()

    def _natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def _refresh(self):  # Call upon upload
        self.emb_len = len(os.listdir(self.emb_root))
        self.mapping = self._load_mapping()
        self.all_emb = self._load_emb()
        self.reverse_map = self._get_reverse_map()

    def _get_reverse_map(self):
        reverse_map = {}
        for emb_file in self.mapping:
            file_name = self.mapping[emb_file]["file_name"]
            index = self.mapping[emb_file]["index"]
            if not file_name in reverse_map:
                reverse_map[file_name] = {}
            reverse_map[file_name][str(index)] = emb_file
        return reverse_map

    def _load_emb(self):
        emb_list = []
        emb_files = os.listdir(self.emb_root)
        emb_files = self._natural_sort(emb_files)
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

    def compare(self, img_file, face_index, k):
        emb_file = self.reverse_map[img_file][face_index]
        emb_tensor = torch.load(f"{self.emb_root}/{emb_file}")[0]
        emb_tensor = emb_tensor/torch.linalg.norm(emb_tensor)  # Normalisation
        norm_emb = torch.transpose(self.all_emb, 0, 1)
        print(emb_tensor.size(), norm_emb.size())
        cos_sim = torch.matmul(emb_tensor, norm_emb)
        print(cos_sim)
        # Plus 1 so that it does not count itself
        k += 1
        topk_cos = torch.topk(cos_sim, k)
        conf = topk_cos.values.tolist()[1:]
        topk_index = topk_cos.indices.tolist()[1:]
        print(topk_cos, topk_index)
        similar_entity_files = [self.mapping[f"{i}.pt"] for i in topk_index]

        return similar_entity_files, conf


EmbMgr = M2E2EmbMgr()


class PopulateFaceEmb(BaseModel):
    face: List[float]
    file_name: str
    index: int


class SourceFaceEmb(BaseModel):
    img_file: str
    face_index: int
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
    img_file = image_data['img_file']
    face_index = image_data['face_index']
    k = image_data['k']

    entities, conf = EmbMgr.compare(
        img_file, str(face_index), k)  # Top 5 results

    response = {}
    response['files'] = entities
    response['similarity'] = conf
    return response


@ api.get("/reverse_map/")
def reverse_map():
    return EmbMgr.reverse_map


@ api.get("/emb/")
def embs():
    print(EmbMgr.all_emb)
    return EmbMgr.all_emb
