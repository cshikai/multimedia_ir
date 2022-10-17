from inference_api.visual_attention.manager import VAManager
from inference_api.visual_attention.log_manager import VALogDatabaseManager

from elasticsearch import Elasticsearch

ELASTIC_URL = "https://elasticsearch:9200"
client = Elasticsearch(ELASTIC_URL,
                       basic_auth=('elastic', 'changeme'),
                       verify_certs=False
                       )

log_manager = VALogDatabaseManager('visual_attention_caption')
va_manager = VAManager(log_manager)


def download_image_b64(server_path):
    body = {'server_path': server_path}
    r = requests.get('http://image_server:8000/download/', json=body)
    return r.json()['image']


def ledger():
    pass


def rollback():
    pass


if __name__ == "__main__":
    merge_type_list, text_entity_id_list, visual_entity_id_list = va_manager.infer([
                                                                                   article_id])

# To return: bbox,
