from haystack.document_stores import ElasticsearchDocumentStore

import pandas as pd
import ast
import json 

from tqdm import tqdm

document_store = ElasticsearchDocumentStore(host="elasticsearch", port="9200", username="elastic", password="changeme", scheme="https", verify_certs=False, index="documents",search_fields=['content','title'])

df = pd.read_csv('/data/articles.csv')
print(df.iloc[0])
docs = []

for idx, row in df.head(10).iterrows():
    doc = {}
    doc['content'] = row['text']
    doc['meta'] = {'ID':row['doc_ID'],'query':row['query'],'link':row['link'],'images':row['images'],'image_captions':row['image_captions']}
    docs.append(doc)

document_store.write_documents(docs)
