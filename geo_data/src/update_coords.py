from elasticsearch import Elasticsearch, helpers
import yaml


def read_yaml(file_path='config.yaml'):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = read_yaml()

ELASTIC_URL = config['ELASTICSEARCH']['URL']
INDEX_NAME = 'documents'
ELASTIC_USERNAME = config['ELASTICSEARCH']['ELASTIC_USERNAME']
ELASTIC_PASSWORD = config['ELASTICSEARCH']['ELASTIC_PASSWORD']

client = Elasticsearch(ELASTIC_URL,  # ca_certs="",
                       verify_certs=False,
                       basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))


def get_location(loc_name):
    geo_query = {
        "function_score": {
            "query": {
                "multi_match": {
                    "query": loc_name,
                    # Emphasis on preferred name > original name > alternate name; score boosting
                    "fields": ["asciiname^2", "alternatenames", "preferrednames^2.5"]
                }
            },
            "script_score": {
                "script": {
                    # Scale feature_class_num from 1-9 to 0.7 to 1, serving as a multiplier on the score.
                    "source": "(doc['feature_class_num'].value-1)*0.0375+0.7"
                }
            }
        }
    }
    geo_resp = client.search(index='geonames', query=geo_query)
    return geo_resp


count = 0
if __name__ == "__main__":
    # Size 1000 because got only 700+ documents
    resp = client.search(index=INDEX_NAME, query={"match_all": {}}, size=1000)
    print("Got %d Hits:" % resp['hits']['total']['value'])
    for hit in resp['hits']['hits']:
        loc_list = []
        for entity in hit['_source']['text_entities']:
            if entity['mention_type'] == 'LOC' or entity['mention_type'] == 'GPE':
                # No entity linked
                entity_name = entity['entity_name']
                if entity_name == 'Unknown':
                    continue
                print(entity_name)
                geo_resp = get_location(entity_name)

                # If no location found
                if geo_resp['hits']['total']['value'] == 0:
                    print("{}: No location found!!".format(
                        entity_name))
                    continue
                latitude = geo_resp['hits']['hits'][0]['_source']['latitude']
                longitude = geo_resp['hits']['hits'][0]['_source']['longitude']
                print("{}: FOUND: {}".format(entity_name,
                      geo_resp['hits']['hits'][0]['_source']['name']))

                loc_list.append({'entity_name': entity_name,
                                 'latitude': latitude,
                                 'longitude': longitude
                                 })
        q = {
            "script": {
                "source": "ctx._source.geo_data=params.data",
                "params": {
                    "data": loc_list
                },
                "lang": "painless"
            },
            "query": {
                "match": {
                    "_id": hit['_id']
                }
            }
        }
        client.update_by_query(
            body=q, index=INDEX_NAME)
        # if count == 10:
        #     break
        # count += 1
