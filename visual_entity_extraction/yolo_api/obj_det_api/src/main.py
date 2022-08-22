from yolo.manager import YOLOManager
# from embeddings.uploader import Uploader
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

yolo = YOLOManager()

class Image(BaseModel):
    image: str

api = FastAPI()

@api.post("/infer")
def infer(img_data: Image):
    """
    Takes in an Image object and predicts the id of the face based
    on existing list of embeddings; 
    Returns the id, confidence and the corresponding bounding box. 
    """
    img_dict = img_data.dict()
    # forward dict to YOLO

    bbox_list, emb_list = yolo.detect(img_dict)
    ret_dict = {
        "bboxes": bbox_list,
        "embs": emb_list  
    }
    return ret_dict
