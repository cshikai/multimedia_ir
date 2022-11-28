from sys import exc_info
import requests
import json
from typing import List, Dict
import pandas as pd
import ast
import re
import time
import tqdm

from haystack.document_stores import ElasticsearchDocumentStore


def predict_jerex(dataset: Dict):
    response = requests.post('http://0.0.0.0:8080/df_link', json=dataset)
    response = response.json()
    df = pd.json_normalize(response, max_level=0)

    print(df.head())
    print(df.info())

    df.to_csv("data/test_jerex.csv", index=False)

    return df


def predict_blink(dataset: Dict):

    response = requests.post('http://blink:5050/df_link', json=dataset)

    df = pd.json_normalize(response.json(), max_level=0)

    print(df)
    print(df.info())

    df.to_csv("data/articles_entity_linked.csv", index=False)

    return df


def generate_entity_linking_df_entities(results_df):

    entities_linking_df = pd.DataFrame(columns=[
                                       'doc_id', 'mention', 'mention_span', 'char_spans', 'mention_type', 'context_left', 'context_right'])

    for idx, row in results_df.iterrows():
        doc_id = row['doc_id']
        if type(row['entities']) == str:
            entities_rows = ast.literal_eval(row['entities'])
            tokens = ast.literal_eval(row['tokens'])
        else:
            entities_rows = row['entities']
            tokens = row['tokens']
        entities = []
        for entity in entities_rows:
            if len(entity['entity_names']) > 0:
                head_entity = max(entity['entity_names'])
                if head_entity not in entities:
                    # print("Head Entity:")
                    # print(head_entity)
                    entity_idx = entity['entity_names'].index(
                        max(entity['entity_names']))
                    entity_span = entity['entity_spans'][entity_idx]
                    entity_char_span = entity['char_spans'][entity_idx]
                    left_context = " ".join(
                        tokens[entity_span[0]-100:entity_span[0]])
                    left_context = re.sub(r"\S*https?:\S*", '', left_context)
                    # print("Left context: ",left_context)
                    # print("\n")
                    right_context = " ".join(
                        tokens[entity_span[1]:entity_span[1]+100])
                    right_context = re.sub(r"\S*https?:\S*", '', right_context)
                    # print("Right context: ",right_context)
                    # print("\n")
                    entities_linking_df.loc[-1] = [doc_id, head_entity, tuple(
                        entity_span), entity_char_span, entity['entity_type'], left_context, right_context]  # adding a row
                    entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                    entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                    entities.append(head_entity)

    # print(entities_linking_df.head())
    return entities_linking_df


if __name__ == '__main__':

    start = time.time()

    document_store = ElasticsearchDocumentStore(host="elasticsearch",
                                                port="9200",
                                                username="elastic",
                                                password="changeme",
                                                scheme="https",
                                                verify_certs=False,
                                                index='documents_m2e2',
                                                search_fields=['content', 'title'])

    documents = document_store.get_all_documents()

    articles_df = pd.DataFrame(
        columns=['ID', 'elasticsearch_ID', 'title', 'image_captions', 'images'])

    for document in documents:
        articles_df.loc[-1] = [document.meta['ID'], document.id, document.meta['link'],
                               document.meta['image_captions'], document.meta['images']]  # adding a row
        articles_df.index = articles_df.index + 1  # shifting index
        articles_df = articles_df.sort_index()  # sorting by index

    # articles_df = pd.read_csv(
    #     'data/m2e2.csv')
    print(articles_df.info())
    print(articles_df.head())

    exploded_df = pd.DataFrame(
        columns=['doc_ID', 'elasticsearch_ID', 'title', 'text', 'ID'])

    for idx, row in articles_df.iterrows():
        doc_id = row['ID']
        es_id = row['elasticsearch_ID']
        title = row['title']
        for idx in range(0, len(row['images'])):
            exploded_df.loc[-1] = [doc_id, es_id, title,
                                   row['image_captions'][idx], row['images'][idx]]  # adding a row
            exploded_df.index = exploded_df.index + 1  # shifting index
            exploded_df = exploded_df.sort_index()  # sorting by index

    exploded_df['ID'] = exploded_df['ID'].apply(
        lambda string: string.split(".")[-2].replace("/", ""))

    print(exploded_df.info())
    print(exploded_df.head())

    # df_json = exploded_df.to_json(orient="records")
    # df_json = json.loads(df_json)
    # jerex_results = predict_jerex(df_json)
    # print("jerex results: ", jerex_results)

    # jerex_results = pd.read_csv('data/test_jerex.csv')
    # jerex_results = jerex_results[:10]
    # jerex_infered = jerex_results[jerex_results.relations != '[]']
    # print("relations: ")
    # print(jerex_infered.info())

    # jerex_infered = jerex_results[jerex_results.entities != '[]']
    # jerex_infered = jerex_results
    # print("entities: ")
    # print(jerex_infered.info())

    # count = 0
    # for idx, row in jerex_infered.iterrows():
    #     if type(row['entities']) == str:
    #         entities = ast.literal_eval(row['entities'])
    #     else:
    #         entities = row['entities']

    #     for entity_row in entities: 
    #         count += len(entity_row)

    # print("total number of entities: ", count)
    # print("Avg count of entities: ", count/len(jerex_infered))

    # entity_linking_df = generate_entity_linking_df_entities(jerex_results)
    # entity_linking_df = entity_linking_df[entity_linking_df.mention_type != 'TIME']
    # entity_linking_df = entity_linking_df[entity_linking_df.mention_type != 'NUM']
    # print(entity_linking_df.info())

    # # entity_linking_df = pd.read_csv('data/articles_entity_linked.csv')
    # # entity_linking_df = entity_linking_df[entity_linking_df.entity_names != 'Unknown']
    # # print(entity_linking_df.info())

    # print(entity_linking_df)
    # entity_linking_df.to_csv("data/entity_linking_df.csv", index=False)
    # # entity_linking_df = pd.read_csv('/home/shearman/Desktop/work/BLINK_es/data/entity_linking_df.csv')
    # # entity_linking_df =entity_linking_df.iloc[:10,:]

    # df_json = entity_linking_df.to_json(orient="records")
    # df_json = json.loads(df_json)

    # blink_results = predict_blink(df_json)

    # blink_results = pd.read_csv('data/articles_entity_linked.csv')
    # blink_results = blink_results[blink_results['mention'].notna()]
    # blink_results.loc[blink_results["entity_link"]
    #                   == "Unknown", "entity_link"] = str(-1)
    # # blink_results["entity_link"] = pd.to_numeric(blink_results["entity_link"])
    # blink_results['doc_id'] = blink_results.doc_id.astype(str)
    # # blink_results['doc_id'] = blink_results['doc_id'].apply(
    # #     lambda string: "/images/M2E2/VOA_EN_NW_2016.05.20." + string[:-1] + "/" + string[-1] + ".h5")

    # list_of_cluster_dfs = blink_results.groupby('doc_id')

    # entities = []
    # ids = []
    # image_ids = []

    # for group, cluster_df in list_of_cluster_dfs:
    #     doc_entities = []
    #     doc_id = cluster_df['doc_id'].tolist()[0]
    #     image_id = doc_id[:-1] + "/" + doc_id[-1]
    #     mentions = cluster_df['mention'].tolist()
    #     cluster_df['char_spans'] = cluster_df['char_spans'].apply(
    #         lambda list_string: ast.literal_eval(list_string))
    #     mention_spans = cluster_df['char_spans'].tolist()
    #     mention_spans = [tuple(span) for span in mention_spans]
    #     mentions_type = cluster_df['mention_type'].tolist()
    #     entity_links = cluster_df['entity_link'].tolist()
    #     entity_names = cluster_df['entity_names'].tolist()

    #     for idx in range(0, len(mentions)):
    #         mention = dict()
    #         mention['mention'] = mentions[idx]
    #         mention['mention_type'] = mentions_type[idx]
    #         mention['mention_span'] = mention_spans[idx]
    #         mention['entity_link'] = str(entity_links[idx])
    #         mention['entity_name'] = entity_names[idx]
    #         doc_entities.append(mention)
    #     ids.append(doc_id)
    #     image_ids.append(image_id)
    #     entities.append(doc_entities)

    # entities_df = pd.DataFrame()
    # entities_df['ID'] = ids
    # entities_df['identified_entities'] = entities
    # entities_df['image_ID'] = image_ids

    # print(entities_df.head())

    # results_df = pd.merge(exploded_df, entities_df, how="left", on=["ID"])

    # template_mention = dict()
    # template_mention['mention'] = ""
    # template_mention['mention_type'] = ""
    # template_mention['mention_span'] = []
    # template_mention['entity_link'] = ""
    # template_mention['entity_name'] = ""
    # template_entity = [mention]

    # # results_df = results_df.fillna(value={"identified_entities": []})

    # print(results_df.info())
    # results_df.to_csv("data/jerex_plus_blink.csv", index=False)

    # results_df = results_df.fillna(value={"image_ID": ""})

    # list_of_cluster_dfs = results_df.groupby('doc_ID')

    # entities = []
    # ids = []

    # for group, cluster_df in list_of_cluster_dfs:
    #     doc_entities = []
    #     doc_id = cluster_df['doc_ID'].tolist()[0]
    #     images_entities = cluster_df['identified_entities'].tolist()
    #     images = cluster_df['image_ID'].tolist()
    #     IDs = cluster_df['ID'].tolist()

    #     for idx in range(0, len(images)):
    #         if images[idx] != '':
    #             image_mentions = {}
    #             image_mentions['file_name'] = images[idx]
    #             mentions = []
    #             mention_types = []
    #             mention_spans = []
    #             entity_links = []
    #             entity_names = []

    #             for mention_dict in images_entities[idx]:
    #                 mentions.append(mention_dict['mention'])
    #                 mention_types.append(mention_dict['mention_type'])
    #                 mention_spans.append(mention_dict['mention_span'])
    #                 entity_links.append(mention_dict['entity_link'])
    #                 entity_names.append(mention_dict['entity_name'])

    #             image_mentions['mentions'] = mentions
    #             image_mentions['mention_types'] = mention_types
    #             image_mentions['mention_spans'] = mention_spans
    #             image_mentions['entity_links'] = entity_links
    #             image_mentions['entity_names'] = entity_names

    #             doc_entities.append(image_mentions)
    #         # else:
    #         #     doc_entities[IDs[idx][:-1] + "/" + IDs[idx][-1]] = []
    #         #     print(IDs[idx][:-1] + "/" + IDs[idx][-1])

    #     if len(doc_entities) != 0:
    #         ids.append(doc_id)
    #         entities.append(doc_entities)

    # entities_df = pd.DataFrame()
    # entities_df['ID'] = ids
    # entities_df['identified_entities'] = entities

    # print(entities_df.head())

    # results_df = pd.merge(articles_df, entities_df, how="left", on=["ID"])

    # print(results_df.info())
    # results_df.to_csv("data/images_captions.csv", index=False)

    captions_df = pd.read_csv("data/images_captions.csv")
    captions_df['identified_entities'] = captions_df['identified_entities'].apply(
        lambda list_string: ast.literal_eval(list_string))
    captions_df['images'] = captions_df['images'].apply(
        lambda list_string: ast.literal_eval(list_string))
    results_df = pd.DataFrame(
        columns=list(captions_df.columns))

    for idx, row in captions_df.iterrows():
        images_ids = row['images']
        entities = row['identified_entities']

        new_entities = []
        for entity in entities:
            print(entity)
            sub_filename = entity['file_name']
            for id in images_ids:
                if sub_filename in id:
                    real_filename = id
            print(real_filename)
            entity.update({"file_name": real_filename})
            new_entities.append(entity)
        results_df.loc[-1] = [row['ID'], row['elasticsearch_ID'], row['title'],
                              row['image_captions'], row['images'], entities]  # adding a row
        results_df.index = results_df.index + 1  # shifting index
        results_df = results_df.sort_index()  # sorting by index

    # Update results to ElasicSearch
    for idx, row in results_df.iterrows():
        print(row['elasticsearch_ID'])
        print(row['identified_entities'])
        meta_dict = {'text_caption_entities': row['identified_entities']}
        document_store.update_document_meta(
            id=row['elasticsearch_ID'], meta=meta_dict)

    end = time.time()
    print("Time to complete jerex and entity linking", end - start)
