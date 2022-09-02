import base64
import json
from chardet import detect
import requests
import yaml
import os

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
        detection_dict = {}
        for img_file in os.listdir(img_folder+'/'+subfolder):
            detection_dict[img_file] = {}
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


            detection_dict[img_file]['person_bbox'] = res_fn['bb']
            detection_dict[img_file]['person_id'] = res_fn['cos_id']
            detection_dict[img_file]['person_conf'] = res_fn['cos_conf']

            detection_dict[img_file]['obj_bbox'] = res_yolo['bbox']
            detection_dict[img_file]['obj_class'] = res_yolo['classes']
            detection_dict[img_file]['obj_conf'] = res_yolo['conf']

        # Convert dict to str, so as to retain shape when uploaded to ES
        detection_dict_str = json.dumps(detection_dict)
        q = {
            "script": {
                "source": "ctx._source.name=params.infer",
                "params": {
                    "infer": detection_dict_str
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
