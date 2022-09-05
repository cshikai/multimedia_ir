from yolo.manager import YOLOManager

import yaml
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()
coco_classes = config['names']
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
    pred_list, emb_list = yolo.detect(img_dict)
    # res_yolo['bboxes'] format: [x1 y1 x2 y2 conf class]
    if len(pred_list) == 0:
        obj_bboxes = []
        obj_classes = []
        obj_conf = []
    else:
        obj_bboxes = [[int(element) for element in sublist[:4]]
                      for sublist in pred_list[0]]
        obj_classes = [int(sublist[-1]) for sublist in pred_list[0]]
        obj_conf = [sublist[-2] for sublist in pred_list[0]]
    if config['resolve_class']:
        ret_dict = {
            "bbox": obj_bboxes,
            "classes": [coco_classes[i] for i in obj_classes],
            "conf": obj_conf,
            "embs": emb_list
        }
    else:
        ret_dict = {
            "bbox": obj_bboxes,
            "classes": obj_classes,
            "conf": obj_conf,
            "embs": emb_list
        }
    return ret_dict
