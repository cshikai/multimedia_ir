import sys
sys.path.append("./yolo")

import torch
import yaml
import numpy as np
from imageio.v2 import imread
import cv2
import base64
from numpy import random
from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer, get_embedding)
from utils.datasets import letterbox
from utils.torch_utils import select_device, load_classifier



def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()


class YOLOManager():

    def __init__(self):
        device = config['device']
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(
            config['weights'], map_location=self.device)  # Load YOLO Model
        self.imgsz = check_img_size(
            config['img_size'], s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16
        if config['emb']:
            # Load pre-saved resnet model locally, so don't need to download each time container spins up
            self.classify_model = load_classifier(
                name='resnet101', n=2, local='resnet101.pt')
            self.classify_model.to(self.device).eval()
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(len(self.names))]

        # model warm up
        img = torch.zeros((1, 3, self.imgsz, self.imgsz),
                          device=self.device)  # init img
        _ = self.model(img.half(
        ) if self.half else img) if self.device.type != 'cpu' else None  # run once

    def detect(self, img_b64):
        bbox_list = []
        emb_list = []
        img_bytes = base64.b64decode(img_b64['image'].encode('utf-8'))
        img = imread(img_bytes, pilmode='RGB')

        # From datasets.py LoadImages
        # img0 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Conversion not required
        img0 = img
        img = letterbox(img0, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        #
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Predict
        pred = self.model(img, augment=config['augment'])[0]
        # NMS
        pred = non_max_suppression(
            pred, config['conf_thres'], config['iou_thres'], classes=None, agnostic=config['agnostic_nms'])

        img_embs = []
        if config['emb']:
            img_embs = get_embedding(pred, self.classify_model, img, img0)
            emb_list.append(img_embs.cpu().tolist())

        # Why need to scale coords ah? Because image is scaled to 640 during pred
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0.shape).round()
                bbox_list.append(det.cpu().tolist())
        return bbox_list, emb_list
