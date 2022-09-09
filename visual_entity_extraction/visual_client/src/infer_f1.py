import base64
import json
import requests
import yaml
import os
import numpy as np

from PIL import Image, ImageDraw
from elasticsearch import Elasticsearch


def read_yaml(file_path='config_f1.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()

ELASTIC_URL = config['ELASTICSEARCH']['URL']
INDEX_NAME = config['ELASTICSEARCH']['INDEX_NAME']
ELASTIC_USERNAME = config['ELASTICSEARCH']['ELASTIC_USERNAME']
ELASTIC_PASSWORD = config['ELASTICSEARCH']['ELASTIC_PASSWORD']

client = Elasticsearch(ELASTIC_URL,  # ca_certs="",
                       verify_certs=False,
                       basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))


if __name__ == '__main__':

    img_folder = config['infer']['img_folder']
    for subfolder in os.listdir(img_folder):
        print(subfolder)
        inference_list = []

        for img_file in os.listdir(img_folder+'/'+subfolder):
            detection_dict = {}
            file_index = img_file.split('_')[1].split('.')[0]
            file_key = '/images/f1/{}/{}.h5'.format(subfolder, file_index)
            with open(img_folder+'/'+subfolder+'/'+img_file, "rb") as f:
                im_bytes = f.read()
            im_b64 = base64.b64encode(im_bytes).decode("utf8")

            headers = {'Content-type': 'application/json',
                       'Accept': 'text/plain'}
            payload = json.dumps({"image": im_b64})

            r_fn = requests.post(
                '{}/infer'.format(config['endpt']['fn_endpt']), data=payload, headers=headers)
            res_fn = json.loads(r_fn.text)

            r_yolo = requests.post(
                '{}/infer'.format(config['endpt']['yolo_endpt']), data=payload, headers=headers)
            res_yolo = json.loads(r_yolo.text)

            detection_dict['file_name']=file_key

            # Facenet Inference
            mask = [True if a > 0.5 else False for a in res_fn['cos_conf']]
            id_list = [a if mask[res_fn['cos_id'].index(a)] else -1 for a in res_fn['cos_id']]
            res_fn['bb'] = [[max(b, 0) for b in a] for a in res_fn['bb']] # Clip negative value to 0
            detection_dict['person_bbox'] = res_fn['bb']
            detection_dict['person_id'] = id_list
            detection_dict['person_conf'] = res_fn['cos_conf']

            # YOLO Inference
            mask = [True if a > 0.5 else False for a in res_yolo['conf']]
            obj_list = [a if mask[res_yolo['classes'].index(a)] else 'Unknown' for a in res_yolo['classes']]
            res_yolo['bbox'] = [[max(b, 0) for b in a] for a in res_yolo['bbox']]
            detection_dict['obj_bbox'] = res_yolo['bbox']
            detection_dict['obj_class'] = obj_list
            detection_dict['obj_conf'] = res_yolo['conf']

            inference_list.append(detection_dict)

        q = {
            "script": {
                "source": "ctx._source.visual_entities=params.infer",
                "params": {
                    "infer": inference_list
                },
                "lang": "painless"
            },
            "query": {
                "match": {
                    "ID": subfolder
                }
            }
        }
        client.update_by_query(
            body=q, index=config['ELASTICSEARCH']['INDEX_NAME'])
