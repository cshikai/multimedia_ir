from haystack.document_stores import ElasticsearchDocumentStore

import pandas as pd
import ast
import json
from image_uploader import F1ImageUploader

from tqdm import tqdm

uploader = F1ImageUploader()
document_store = ElasticsearchDocumentStore(host="elasticsearch", port="9200", username="elastic", password="changeme",
                                            scheme="https", verify_certs=False, index="documents_m2e2", search_fields=['content', 'title'])


def convert_to_list(string):
    return string.strip('[]').replace("'", "").split(',')


df = pd.read_csv('/data/m2e2.csv')
# df = pd.read_csv('/data/articles.csv')

df['images'] = df['images'].apply(lambda images: ast.literal_eval(images))
df['image_captions'] = df['image_captions'].apply(
    lambda image_captions: ast.literal_eval(image_captions))
print(df.iloc[0])
docs = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    doc = {}
    doc['content'] = row['text']

    #
    server_paths = []

    for image_filename in row['images']:
        print(image_filename)
        server_path = uploader.upload_single_image_b64(image_filename)
        # server_paths.append(server_path)

    # doc['meta'] = {'ID': row['doc_ID'], 'query': row['query'], 'link': row['link'],
    #                'images': server_paths, 'image_captions': row['image_captions']}

    # doc['meta'] = {'ID': row['doc_ID'], 'link': row['url'],
    #                'images': server_paths, 'image_captions': row['image_captions'], 'timestamp': row['timestamp']}
    # docs.append(doc)

# document_store.write_documents(docs)
