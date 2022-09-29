from embeddings.embeddings import EmbeddingDataset
from embeddings.uploader import Uploader
# from embeddings.identify import Identify

import yaml
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()

EMBEDDING_PATH = config['EMB']['emb_path']


dataset = EmbeddingDataset(EMBEDDING_PATH, local=False)
uploader = Uploader(EMBEDDING_PATH, local=False)


class Image(BaseModel):
    image: str


class ImageFolder(BaseModel):
    id: int
    images: List[Image]


api = FastAPI()


@api.post("/infer")
def infer(img_data: Image):
    """
    Takes in an Image object and predicts the id of the face based
    on existing list of embeddings; 
    Returns the id, confidence and the corresponding bounding box. 
    """
    img_dict = img_data.dict()

    all_cos_id, all_cos_conf, all_euc_id, all_euc_conf, bb, all_embs = dataset.find_id(
        img_dict)
    if config['EMB']['get_emb']:
        ret_dict = {
            "cos_id": all_cos_id,
            "cos_conf": all_cos_conf,
            "euc_id": all_euc_id,
            "euc_conf": all_euc_conf,
            "bb": bb,
            "emb": all_embs
        }
    else:
        ret_dict = {
            "cos_id": all_cos_id,
            "cos_conf": all_cos_conf,
            "euc_id": all_euc_id,
            "euc_conf": all_euc_conf,
            "bb": bb
        }
    return ret_dict


@api.post("/generate")
def generate(img_list: ImageFolder):
    """
    Takes in an ImageFolder object, generates an embedding and saves it locally.
    """
    img_dict = img_list.dict()
    emb = dataset.generate_embedding_from_b64(
        img_dict['id'], img_dict['images'])
    if emb == []:
        return {"Response": "No embedding generated"}
    else:
        uploader.save_emb(img_dict['id'], emb)
        return {"Response": "OK"}
