import torch
import os
import yaml
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()
ELASTIC_URL = config['ELASTICSEARCH']['URL']
INDEX_NAME = config['ELASTICSEARCH']['INDEX_NAME']
ELASTIC_USERNAME = config['ELASTICSEARCH']['ELASTIC_USERNAME']
ELASTIC_PASSWORD = config['ELASTICSEARCH']['ELASTIC_PASSWORD']


class Uploader():

    def __init__(self, path, local):
        self.local = local
        if self.local:
            self.path = path
        else:
            self.client = Elasticsearch(ELASTIC_URL,
                                        # ca_certs="",
                                        verify_certs=False,
                                        basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))

    def save_emb(self, id, emb):
        if self.local:
            path = os.path.join(self.path, '')
            torch.save(emb, path+str(id)+'.pt')
            print("Saved as", path+str(id)+'.pt')
        else:
            request = {}
            request["_id"] = id
            request["_index"] = INDEX_NAME
            request["_source"] = {}
            request["_source"]["script"] = {
                "source": "ctx._source.face_emb = params.vector",
                "params": {
                    "vector": emb.numpy()
                }
            }
            request["_op_type"] = 'update'

            bulk(self.client, [request])
            self.client.indices.refresh(index=INDEX_NAME)
        return
