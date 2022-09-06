import requests
import json
from typing import List, Dict
import pandas as pd
import ast
import re
import time
import tqdm

from haystack.document_stores import ElasticsearchDocumentStore

if __name__ == '__main__':

    start = time.time()

    document_store = ElasticsearchDocumentStore(host="elasticsearch",
                                                port="9200",
                                                username="elastic",
                                                password="changeme",
                                                scheme="https",
                                                verify_certs=False,
                                                index='documents',
                                                search_fields=['content', 'title'])

    documents = document_store.get_all_documents()

    articles_df = pd.DataFrame(
        columns=['ID', 'title', 'text', 'elasticsearch_ID'])

    for document in documents:
        articles_df.loc[-1] = [document.meta['ID'], document.meta['link'],
                               document.content, document.id]  # adding a row
        articles_df.index = articles_df.index + 1  # shifting index
        articles_df = articles_df.sort_index()  # sorting by index

    print(articles_df.info())
    print(articles_df.head())

    timestamp_df = pd.read_csv(
        "data/articles_timestamped.csv")
    timestamp_df = timestamp_df.rename(columns={"doc_ID": "ID"})

    results_df = pd.merge(articles_df, timestamp_df, on=["ID"])

    # Update results to ElasicSearch
    for idx, row in results_df.iterrows():
        meta_dict = {'timestamp': row['es_datetime']}
        document_store.update_document_meta(
            id=row['elasticsearch_ID'], meta=meta_dict)

    end = time.time()
    print("Time to complete jerex and entity linking", end - start)
