from cgitb import text
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

    df.to_csv("data/text_test_jerex.csv", index=False)

    return df


def predict_blink(dataset: Dict):

    response = requests.post('http://blink:5050/df_link', json=dataset)

    df = pd.json_normalize(response.json(), max_level=0)

    print(df)
    print(df.info())

    df.to_csv("data/text_articles_entity_linked.csv", index=False)

    return df


def generate_entity_linking_df(results_df):

    entities_linking_df = pd.DataFrame(
        columns=['doc_id', 'mention', 'mention_type', 'context_left', 'context_right'])

    for idx, row in results_df.iterrows():
        doc_id = row['doc_id']
        if type(row['relations']) == str:
            relations = ast.literal_eval(row['relations'])
            tokens = ast.literal_eval(row['tokens'])
        else:
            relations = row['relations']
            tokens = row['tokens']
        entities = []
        for relation in relations:
            head_entity = " ".join(
                tokens[relation['head_span'][0]:relation['head_span'][1]])
            if head_entity not in entities:
                print("Head Entity:")
                print(head_entity)
                left_context = " ".join(
                    tokens[relation['head_span'][0]-100:relation['head_span'][0]])
                left_context = re.sub(r"\S*https?:\S*", '', left_context)
                print("Left context: ", left_context)
                print("\n")
                right_context = " ".join(
                    tokens[relation['head_span'][1]:relation['head_span'][1]+100])
                right_context = re.sub(r"\S*https?:\S*", '', right_context)
                print("Right context: ", right_context)
                print("\n")
                entities_linking_df.loc[-1] = [doc_id, head_entity,
                                               relation['head_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(head_entity)

            tail_entity = " ".join(
                tokens[relation['tail_span'][0]:relation['tail_span'][1]])
            if tail_entity not in entities:
                print("Tail Entity:")
                print(tail_entity)
                left_context = " ".join(
                    tokens[relation['tail_span'][0]-100:relation['tail_span'][0]])
                left_context = re.sub(r"\S*https?:\S*", '', left_context)
                print("Left context: ", left_context)
                print("\n")
                right_context = " ".join(
                    tokens[relation['tail_span'][1]:relation['tail_span'][1]+100])
                right_context = re.sub(r"\S*https?:\S*", '', right_context)
                print("Right context: ", right_context)
                print("\n")
                entities_linking_df.loc[-1] = [doc_id, tail_entity,
                                               relation['tail_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(tail_entity)

    print(entities_linking_df.head())
    return entities_linking_df


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
        columns=['ID', 'title', 'text', 'elasticsearch_ID'])

    for document in documents:
        articles_df.loc[-1] = [document.meta['ID'], document.meta['link'],
                               document.content, document.id]  # adding a row
        articles_df.index = articles_df.index + 1  # shifting index
        articles_df = articles_df.sort_index()  # sorting by index

    # articles_df = pd.read_csv(
    #     'data/m2e2.csv')
    print(articles_df.info())
    print(articles_df.head())
    # articles_df = articles_df[:10]

    # df_json = articles_df.to_json(orient="records")
    # df_json = json.loads(df_json)
    # jerex_results = predict_jerex(df_json)
    # print("jerex results: ", jerex_results)

    # jerex_results = pd.read_csv('data/test_jerex.csv')
    # jerex_infered = jerex_results[jerex_results.relations != '[]']
    # print("relations: ")
    # print(jerex_infered.info())

    # jerex_infered = jerex_results[jerex_results.entities != '[]']
    # # jerex_infered = jerex_results
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
    # entity_linking_df.to_csv("data/text_entity_linking_df.csv", index=False)
    # # entity_linking_df = pd.read_csv('/home/shearman/Desktop/work/BLINK_es/data/entity_linking_df.csv')
    # # entity_linking_df =entity_linking_df.iloc[:10,:]

    # df_json = entity_linking_df.to_json(orient="records")
    # df_json = json.loads(df_json)

    # blink_results = predict_blink(df_json)

    # blink_results = pd.read_csv('data/text_articles_entity_linked.csv')
    # blink_results = blink_results[blink_results['mention'].notna()]
    # blink_results.loc[blink_results["entity_link"]
    #                   == "Unknown", "entity_link"] = str(-1)
    # blink_results["entity_link"] = pd.to_numeric(blink_results["entity_link"])

    # list_of_cluster_dfs = blink_results.groupby('doc_id')

    # entities = []
    # ids = []

    # for group, cluster_df in list_of_cluster_dfs:
    #     doc_entities = []
    #     doc_id = cluster_df['doc_id'].tolist()[0]
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
    #         mention['entity_link'] = entity_links[idx]
    #         mention['entity_name'] = entity_names[idx]
    #         doc_entities.append(mention)
    #     ids.append(doc_id)
    #     entities.append(doc_entities)

    # entities_df = pd.DataFrame()
    # entities_df['ID'] = ids
    # entities_df['identified_entities'] = entities

    # print(entities_df.head())

    # results_df = pd.merge(articles_df, entities_df, how="left", on=["ID"])

    # template_mention = dict()
    # template_mention['mention'] = ""
    # template_mention['mention_type'] = ""
    # template_mention['mention_span'] = []
    # template_mention['entity_link'] = ""
    # template_mention['entity_name'] = ""
    # template_entity = [mention]

    # results_df = results_df.fillna(value={"identified_entities": []})

    # print(results_df.info())
    # results_df.to_csv("data/jerex_plus_blink.csv", index=False)

    m2e2_df = pd.read_csv("data/m2e2articles_df.csv")

    results_df = articles_df = pd.DataFrame(
        columns=['ID', 'title', 'text', 'elasticsearch_ID', 'text_entities'])

    m2e2_df['text_entities'] = m2e2_df['text_entities'].apply(
        lambda list_string: ast.literal_eval(list_string))

    for idx, row in m2e2_df.iterrows():
        text_entities = []

        for text_entity in row['text_entities']:
            new_entity = {}
            for key, value in text_entity.items():
                if key == 'entity_link':
                    new_entity[key] = str(value)
                    print(new_entity[key])
                else:
                    new_entity[key] = value
            text_entities.append(new_entity)

        results_df.loc[-1] = [row['ID'], row['title'],
                              row['text'], row['elasticsearch_ID'], text_entities]  # adding a row
        results_df.index = results_df.index + 1  # shifting index
        results_df = results_df.sort_index()  # sorting by index

    # Update results to ElasicSearch
    for idx, row in results_df.iterrows():
        meta_dict = {'text_entities': row['text_entities']}
        document_store.update_document_meta(
            id=row['elasticsearch_ID'], meta=meta_dict)

    end = time.time()
    print("Time to complete jerex and entity linking", end - start)
