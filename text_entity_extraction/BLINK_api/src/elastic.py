from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Index, Q, A, Document
from elasticsearch_dsl.query import MultiMatch, Match
from requests.packages import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from utils import load_config

# es = Elasticsearch(
#     "https://localhost:9200",
#     basic_auth=("elastic", "changeme"),
#     verify_certs=False,
# )

es = Elasticsearch("https://elastic:changeme@localhost:9200", verify_certs=False)


def iterate_all_docs(index):
    """
    Get all document from index
    Return list of Hit
    """
    s = Search(using=es, index=index)
    # s_list = [_ for _ in s.scan()]

    return [hit._d_ for hit in s.scan()]


def index_statistics(index):
    """
    Get statistics for an index
    """
    i = Index(using=es, name=index)
    stats = i.stats()
    docs_count = stats["_all"]["total"]["docs"]["count"]
    docs_deleted = stats["_all"]["total"]["docs"]["deleted"]
    store_total_data_set_size_in_bytes = stats["_all"]["total"]["store"][
        "total_data_set_size_in_bytes"
    ]

    return {
        "index": index,
        "docs_count": docs_count,
        "docs_deleted": docs_deleted,
        "store_total_data_set_size_in_bytes": store_total_data_set_size_in_bytes,
    }


def get_docs_count_for_field_name(index, field_name, value):
    """
    # TODO()
    """
    q = Q("multi_match", query=value, fields=[field_name])
    s = Search(using=es, index=index)
    s = s.query(q)
    return s.count()


def get_docs_for_field_name(index, field_name, value):
    """
    # TODO()
    """
    q = Q("multi_match", query=value, fields=[field_name])
    s = Search(using=es, index=index)
    s = s.query(q)
    return [hit._d_ for hit in s.scan()]


def get_unique_list_of_field_name(index, field_name_key):
    """
    Reference: https://stackoverflow.com/questions/29380198/aggregate-a-field-in-elasticsearch-dsl-using-python
    """
    s = Search(using=es, index=index)
    s.aggs.bucket(
        "by_filename",
        "terms",
        field=field_name_key,
    )
    result = s.execute()
    return [each_value.key for each_value in result.aggregations.by_filename.buckets]


# def get_unique_count_for_field_name(index, field_name_value):
#     s = Search(using=es, index=index)
#     s.aggs.metric("by_cluster", "cardinality", field=field_name_value)
#     es_data = s.execute()
#     unique_count = es_data.aggregations.by_cluster.value
#     return unique_count


def update_doc(index, id, doc_dict):
    """
    # TODO()
    """
    doc = Document.get(id=id, using=es, index=index)
    doc.update(test="some-test-value")


# def get_docs_with_filename_id(index, filename, id):
#     """
#     # TODO()
#     """
#     query_body = {
#         "query": {
#             "bool": {"must": [{"term": {"filename": filename}}, {"term": {"ID": id}}]}
#         }
#     }

#     resp = es.search(index=index, body=query_body)
#     print(resp)
#     # print(resp["hits"]["total"]["value"])
#     # if resp["hits"]["total"]["value"] >= 1:
#     #     return True
#     # else:
#     #     return False


def get_docs_with_filename_and_key_value_list(index, filename, key_value_list):
    """
    # TODO()
    """

    # key_value_list = [{"filename": "file.csv"}, {"ID": 1234}]

    # query_body = {
    #     "query": {
    #         "bool": {"must": [{"term": {"filename": filename}}, {"term": {"ID": id}}]}
    #     }
    # }

    query_body = {
        "query": {
            "bool": {"must": [{"term": key_value} for key_value in key_value_list]}
        }
    }

    resp = es.search(index=index, body=query_body)
    if resp["hits"]["total"]["value"] > 0:
        return resp["hits"]["hits"]
    else:
        return ""


def has_doc_with_key_value_list(index, key_value_list):
    """
    Check if filename and ID exists. Return document ID if exist, else empty string
    """
    # query_body = {"query": {"match": {key: value}}}
    # query_body = {"query": {"term": {key + ".keyword": value}}}
    # query_body = {"query": {"term": {key + ".keyword": {"value": value}}}}
    # query_body = {"query": {"term": {key: {"value": value}}}}

    # key_value_list = [{"filename": "file.csv"}, {"ID": 1234}]

    query_body = {
        "query": {
            "bool": {"must": [{"term": key_value} for key_value in key_value_list]}
        }
    }

    resp = es.search(index=index, body=query_body)

    if resp["hits"]["total"]["value"] > 0:
        return resp["hits"]["hits"][0]["_id"]
    else:
        return False


# def has_doc_with_filename_id(index, filename, id):
#     """
#     Check if filename and ID exists. Return document ID if exist, else empty string
#     """
#     query_body = {
#         "query": {
#             "bool": {"must": [{"term": {"filename": filename}}, {"term": {"ID": id}}]}
#         }
#     }

#     resp = es.search(index=index, body=query_body)
#     if resp["hits"]["total"]["value"] > 0:
#         return resp["hits"]["hits"][0]["_id"]
#     else:
#         return ""


def add_docs_to_index(index, filename, docs):
    """
    # TODO()
    """
    for doc in docs:
        if "TYPE" in doc:
            del doc["TYPE"]  # empty
        if "TAGS" in doc:
            del doc["TAGS"]  # empty
        doc["timestamp"] = doc["CREATEDATE"]  # POSTDATE
        doc["filename"] = filename
        resp = es.index(index=index, document=doc)
    es.indices.refresh(index=index)


def delete_docs_from_index(index, filename):
    """
    # TODO()
    """
    s = Search(using=es, index=index).query("match", filename=filename)
    response = s.delete()
    return response


def remove_index(index):
    """
    # TODO()
    """
    es.indices.delete(index=index, ignore=[400, 404])
    # es.options(ignore_status=[400, 404]).indices.delete(index=index)


def create_index(index):
    """
    text, boolean, integer
    """
    config = load_config()

    properties_value = {}
    for k, v in config["required_columns_mapping"].items():
        properties_value[k] = {"type": v}

    mapping = {"mappings": {"properties": properties_value}}

    es.indices.create(
        index=index,
        ignore=400,
        body=mapping,
    )
