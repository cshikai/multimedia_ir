import requests
import json
from typing import List, Dict
import pandas as pd
import ast
import re
import time

from haystack.document_stores import ElasticsearchDocumentStore

def predict_jerex(dataset: Dict):
    response = requests.post('http://0.0.0.0:8080/df_link', json = dataset)
    
    #ipdb.set_trace()
    # print([doc for doc in response.iter_lines()])
    response = response.json()
    # print(type(response))
    # df_json = json.loads(response)
    # print(type(df_json))
    df = pd.json_normalize(response, max_level=0)

    print(df.head())
    print(df.info())

    df.to_csv("data/test_jerex.csv",index=False)

    return df

def predict_blink(dataset: Dict):

    response = requests.post('http://0.0.0.0:5050/df_link', json = dataset)
    # df_json = json.dumps(response.json())
    # df = pd.read_json(df_json, orient="records")

    # print(df.head())
    # print(df.info())

    df = pd.json_normalize(response.json(), max_level=0)

    print(df)
    print(df.info())

    df.to_csv("data/articles_entity_linked.csv",index=False)

    return df

def generate_entity_linking_df(results_df):

    entities_linking_df = pd.DataFrame(columns=['doc_id','mention', 'mention_type','context_left','context_right'])

    for idx, row in results_df.iterrows():
        doc_id = row['doc_id']
        if type(row['relations']) == str:
            relations = ast.literal_eval(row['relations'])
            tokens = ast.literal_eval(row['tokens'])
        else:
            relations =  row['relations']
            tokens = row['tokens']
        entities = []
        for relation in relations:
            head_entity = " ".join(tokens[relation['head_span'][0]:relation['head_span'][1]])
            if head_entity not in entities:
                print("Head Entity:")
                print(head_entity)
                left_context = " ".join(tokens[relation['head_span'][0]-100:relation['head_span'][0]])
                left_context = re.sub(r"\S*https?:\S*", '', left_context)
                print("Left context: ",left_context)
                print("\n")
                right_context = " ".join(tokens[relation['head_span'][1]:relation['head_span'][1]+100])
                right_context = re.sub(r"\S*https?:\S*", '', right_context)
                print("Right context: ",right_context)
                print("\n")
                entities_linking_df.loc[-1] = [doc_id, head_entity, relation['head_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(head_entity)

            tail_entity = " ".join(tokens[relation['tail_span'][0]:relation['tail_span'][1]])
            if tail_entity not in entities:
                print("Tail Entity:")
                print(tail_entity)
                left_context = " ".join(tokens[relation['tail_span'][0]-100:relation['tail_span'][0]])
                left_context = re.sub(r"\S*https?:\S*", '', left_context)
                print("Left context: ",left_context)
                print("\n")
                right_context = " ".join(tokens[relation['tail_span'][1]:relation['tail_span'][1]+100])
                right_context = re.sub(r"\S*https?:\S*", '', right_context)
                print("Right context: ",right_context)
                print("\n")
                entities_linking_df.loc[-1] = [doc_id, tail_entity, relation['tail_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(tail_entity)

    print(entities_linking_df.head())
    return entities_linking_df

def generate_entity_linking_df_entities(results_df):

    entities_linking_df = pd.DataFrame(columns=['doc_id','mention', 'mention_type','context_left','context_right'])

    for idx, row in results_df.iterrows():
        doc_id = row['doc_id']
        if type(row['entities']) == str:
            entities_rows = ast.literal_eval(row['entities'])
            tokens = ast.literal_eval(row['tokens'])
        else:
            entities_rows =  row['entities']
            tokens = row['tokens']
        entities = []
        for entity in entities_rows:
            head_entity = max(entity['entity_names'])
            if head_entity not in entities:
                # print("Head Entity:")
                # print(head_entity)
                entity_idx = entity['entity_names'].index(max(entity['entity_names']))
                entity_span = entity['entity_spans'][entity_idx]
                left_context = " ".join(tokens[entity_span[0]-100:entity_span[0]])
                left_context = re.sub(r"\S*https?:\S*", '', left_context)
                # print("Left context: ",left_context)
                # print("\n")
                right_context = " ".join(tokens[entity_span[1]:entity_span[1]+100])
                right_context = re.sub(r"\S*https?:\S*", '', right_context)
                # print("Right context: ",right_context)
                # print("\n")
                entities_linking_df.loc[-1] = [doc_id, head_entity, entity['entity_type'], left_context, right_context]  # adding a row
                entities_linking_df.index = entities_linking_df.index + 1  # shifting index
                entities_linking_df = entities_linking_df.sort_index()  # sorting by index
                entities.append(head_entity)

    # print(entities_linking_df.head())
    return entities_linking_df


if __name__ == '__main__':

    start = time.time()

    document_store = ElasticsearchDocumentStore(host= "localhost",
                                                port= "9200", 
                                                username= "elastic", 
                                                password= "changeme", 
                                                scheme= "https", 
                                                verify_certs= False, 
                                                index = 'formula1_articles',
                                                search_fields= ['content','title'])

    documents = document_store.get_all_documents()

    articles_df = pd.DataFrame(columns=['ID','title','text','elasticsearch_ID'])

    for document in documents:
        articles_df.loc[-1] = [document.meta['ID'], document.meta['link'],document.content,document.id]  # adding a row
        articles_df.index = articles_df.index + 1  # shifting index
        articles_df = articles_df.sort_index()  # sorting by index

    print(articles_df.info())
    print(articles_df.head())

    # new_entities = [{"text": " The COVID-19 recession is a global economic recession caused by the COVID-19 pandemic. The recession began in most countries in February 2020. After a year of global economic slowdown that saw stagnation of economic growth and consumer activity, the COVID-19 lockdowns and other precautions taken in early 2020 drove the global economy into crisis. Within seven months, every advanced economy had fallen to recession. The first major sign of recession was the 2020 stock market crash, which saw major indices drop 20 to 30%% in late February and March. Recovery began in early April 2020, as of April 2022, the GDP for most major economies has either returned to or exceeded pre-pandemic levels and many market indices recovered or even set new records by late 2020. ", "idx": "https://en.wikipedia.org/wiki?curid=63462234", "title": "COVID-19 recession", "entity": "COVID-19 recession"},
    #  {"text": " The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 (COVID-19) caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The novel virus was first identified from an outbreak in Wuhan, China, in December 2019. Attempts to contain it there failed, allowing the virus to spread worldwide. The World Health Organization (WHO) declared a Public Health Emergency of International Concern on 30 January 2020 and a pandemic on 11 March 2020. As of 15 April 2022, the pandemic had caused more than 502 million cases and 6.19 million deaths, making it one of the deadliest in history. ", "idx": "https://en.wikipedia.org/wiki?curid=62750956", "title": "COVID-19 pandemic", "entity": "COVID-19 pandemic"},
    #  {"text": "Coronavirus disease 2019 (COVID-19) is a contagious disease caused by a virus, the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The first known case was identified in Wuhan, China, in December 2019.[7] The disease quickly spread worldwide, resulting in the COVID-19 pandemic.\
    #   Symptoms of COVID‑19 are variable, but often include fever,[8] cough, headache,[9] fatigue, breathing difficulties, loss of smell, and loss of taste.[10][11][12] Symptoms may begin one to fourteen days after exposure to the virus. At least a third of people who are infected do not develop noticeable symptoms.\
    #   [13] Of those people who develop symptoms noticeable enough to be classed as patients, most (81%) develop mild to moderate symptoms (up to mild pneumonia), while 14% develop severe symptoms (dyspnoea, hypoxia, or more than 50%% lung involvement on imaging), and 5%% develop critical symptoms (respiratory failure, \
    #   shock, or multiorgan dysfunction).[14] Older people are at a higher risk of developing severe symptoms. Some people continue to experience a range of effects (long COVID) for months after recovery, and damage to organs has been observed.[15] Multi-year studies are underway to further investigate the long-term effects of the disease.[15] \
    #   COVID‑19 transmits when people breathe air contaminated by droplets and small airborne particles containing the virus. The risk of breathing these is highest when people are in close proximity, but they can be inhaled over longer distances, particularly indoors. Transmission can also occur if splashed or sprayed with contaminated fluids in the eyes, \
    #   nose or mouth, and, rarely, via contaminated surfaces. People remain contagious for up to 20 days, and can spread the virus even if they do not develop symptoms.[16][17] \
    #   COVID-19 testing methods to detect the virus's nucleic acid include real-time reverse transcription polymerase chain reaction (rRT‑PCR),[18][19] transcription-mediated amplification,[18][19][20] and reverse transcription loop-mediated isothermal amplification (RT‑LAMP)[18][19] from a nasopharyngeal swab.[21]Several COVID-19 vaccines have been approved \
    #   and distributed in various countries, which have initiated mass vaccination campaigns. Other preventive measures include physical or social distancing, quarantining, ventilation of indoor spaces, covering coughs and sneezes, hand washing, and keeping unwashed hands away from the face. The use of face masks or coverings has been recommended in public settings to minimise the risk of transmission. While work is underway to develop drugs that inhibit the virus, the primary treatment is symptomatic. Management involves the treatment of symptoms, supportive care, isolation, and experimental measures.", "idx": "https://en.wikipedia.org/wiki?curid=63030231", "title": "COVID-19", "entity": "COVID-19"}]

    # df_json = json.dumps(new_entities)
    # df_json = json.loads(df_json)
    # response = requests.post('http://0.0.0.0:5050/add_entities', json = df_json)

    df_json = articles_df.to_json(orient="records")
    df_json = json.loads(df_json)
    jerex_results = predict_jerex(df_json)
    print("jerex results: ", jerex_results)

    # jerex_results = pd.read_csv('data/test_jerex.csv')
    # jerex_infered = jerex_results[jerex_results.relations != '[]']
    # print("relations: ")
    # print(jerex_infered.info())

    jerex_infered = jerex_results[jerex_results.entities != '[]']
    print("entities: ")
    print(jerex_infered.info())

    count = 0
    for idx, row in jerex_infered.iterrows():
        if type(row['entities']) == str:
            entities = ast.literal_eval(row['entities'])
        else:
            entities =  row['entities']

        for entity_row in entities:
            count += len(entity_row)

    print("total number of entities: ", count)
    print("Avg count of entities: ", count/len(jerex_infered))


    entity_linking_df = generate_entity_linking_df_entities(jerex_results)
    entity_linking_df = entity_linking_df[entity_linking_df.mention_type != 'TIME']
    entity_linking_df = entity_linking_df[entity_linking_df.mention_type != 'NUM']
    print(entity_linking_df.info())
    # entity_linking_df = pd.read_csv('data/articles_entity_linked.csv')
    # entity_linking_df = entity_linking_df[entity_linking_df.entity_names != 'Unknown']
    # print(entity_linking_df.info())

    # print(entity_linking_df)
    # entity_linking_df.to_csv("data/entity_linking_df.csv",index=False)
    # entity_linking_df = pd.read_csv('/home/shearman/Desktop/work/BLINK_es/data/entity_linking_df.csv')
    # entity_linking_df =entity_linking_df.iloc[:10,:]


    df_json = entity_linking_df.to_json(orient="records")
    df_json = json.loads(df_json)

    blink_results = predict_blink(df_json)

    # df = pd.read_csv('data/articles_entity_linked.csv')

    # list_of_cluster_dfs = df.groupby('doc_id')

    # entities = []
    # ids = []

    # for group, cluster_df in list_of_cluster_dfs:
    #     doc_entities = []
    #     doc_id = cluster_df['doc_id'].tolist()[0]
    #     mentions = cluster_df['mention'].tolist()
    #     mentions_type = cluster_df['mention_type'].tolist()
    #     entity_links = cluster_df['entity_link'].tolist()
    #     entity_names = cluster_df['entity_names'].tolist()
    #     for idx in range(0,len(mentions)):
    #         mention = dict()
    #         mention['mention'] = mentions[idx]
    #         mention['mention_type'] = mentions_type[idx]
    #         mention['entity_link'] = entity_links[idx]
    #         mention['entity_name'] = entity_names[idx]
    #         doc_entities.append(mention)
    #     ids.append(doc_id)
    #     entities.append(doc_entities)

    # entities_df = pd.DataFrame()
    # entities_df['ID'] = ids
    # entities_df['identified_entities'] = entities

    # results_df = pd.merge(articles_df, entities_df, on=["ID"])

    # print(results_df.info())    
    # results_df.to_csv("data/jerex_plus_blink.csv",index=False)

    # Update results to ElasicSearch
    # for idx, row in results_df.iterrows():
    #     meta_dict = {'entities_identified':row['identified_entities']}
    #     document_store.update_document_meta(id=row['elasticsearch_ID'], meta=meta_dict)

    end = time.time()
    print("Time to complete jerex and entity linking",end - start)

