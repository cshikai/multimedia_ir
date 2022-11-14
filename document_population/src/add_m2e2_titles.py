
# WORKS ON elasticsearch==8.3.3 but haystack probably needs older version of ES, which is giving some auth problem :((

import yaml
from tqdm import tqdm
import json

from elasticsearch import Elasticsearch


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()

ELASTIC_URL = config['ELASTICSEARCH']['URL']
INDEX_NAME = config['ELASTICSEARCH']['INDEX_NAME']
ELASTIC_USERNAME = config['ELASTICSEARCH']['ELASTIC_USERNAME']
ELASTIC_PASSWORD = config['ELASTICSEARCH']['ELASTIC_PASSWORD']
TITLE_FILE = config['DATA']['TITLES']

client = Elasticsearch(ELASTIC_URL, ca_certs="",
                       verify_certs=False,
                       basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))

if __name__ == '__main__':

    f = open(TITLE_FILE, "r")
    titles = json.loads(f.read())
    for ID in tqdm(titles):
        q = {
            "script": {
                "source": "ctx._source.title=params.title",
                "params": {
                    "title": titles[ID]
                },
                "lang": "painless"
            },
            "query": {
                "match": {
                    "ID": int(ID)
                }
            }
        }
        client.update_by_query(
            body=q, index=config['ELASTICSEARCH']['INDEX_NAME'])
